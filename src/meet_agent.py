"""
Main integration script for Google Meet voice agent.
Connects Attendee.dev WebSocket bridge with LiveKit voice agent.
"""

import asyncio
import logging
import os
import sys
from typing import Optional
from dotenv import load_dotenv
import numpy as np

from attendee_bridge import AttendeeBridge
from meet_session import MeetSession
from audio_transport import AudioTransport
from livekit_pipeline import LiveKitPipeline

load_dotenv(".env.local")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("meet-agent")


class MeetVoiceAgent:
    """
    Google Meet Voice Agent integrating Attendee.dev with LiveKit.
    
    Architecture:
    Google Meet ←→ Attendee Bot ←→ WebSocket Server ←→ This Agent ←→ LiveKit STT/LLM/TTS
    """

    def __init__(self):
        """Initialize the Meet Voice Agent."""
        self.bridge: Optional[AttendeeBridge] = None
        self.meet_session: Optional[MeetSession] = None
        
        # Audio buffers
        self.audio_buffer = bytearray()
        self.sample_rate = 16000
        
        # LiveKit STT → LLM → TTS pipeline
        self.pipeline = LiveKitPipeline()
        
        # Processing state
        self.is_processing = False
        self.min_speech_duration_ms = 1500  # Minimum 1.5 seconds of audio before processing
        
        logger.info("MeetVoiceAgent initialized with LiveKit STT/LLM/TTS")

    async def audio_from_meet_callback(self, pcm_data: bytes, sample_rate: int):
        """
        Callback when audio is received from Google Meet via Attendee.dev.
        
        Args:
            pcm_data: Raw PCM audio from meeting (16-bit, mono)
            sample_rate: Sample rate of the audio
        """
        duration_ms = AudioTransport.get_audio_duration_ms(pcm_data, sample_rate)
        
        # Log occasionally to avoid spam
        if len(self.audio_buffer) % 50000 < 1000:
            logger.debug(
                f"📥 Received audio from Meet: {len(pcm_data)} bytes, "
                f"{duration_ms:.1f}ms @ {sample_rate}Hz"
            )
        
        # Add to buffer
        self.audio_buffer.extend(pcm_data)
        
        # Process when we have enough audio
        buffer_duration_ms = len(self.audio_buffer) / (sample_rate * 2) * 1000
        
        if buffer_duration_ms >= self.min_speech_duration_ms and not self.is_processing:
            await self.process_audio_chunk()

    async def process_audio_chunk(self):
        """
        Process accumulated audio through LiveKit STT → LLM → TTS pipeline.
        """
        if not self.audio_buffer or self.is_processing:
            return
        
        self.is_processing = True
        
        try:
            # Get all accumulated audio
            chunk = bytes(self.audio_buffer)
            self.audio_buffer = bytearray()
            
            logger.info(f"🔄 Processing {len(chunk)} bytes through LiveKit pipeline...")
            
            # Check if audio contains speech (simple energy check)
            audio_array = np.frombuffer(chunk, dtype=np.int16)
            
            if not self.has_speech(audio_array):
                logger.debug("No speech detected in audio chunk")
                return
            
            logger.info("🎤 Speech detected! Processing through STT → LLM → TTS...")
            
            # Process through full pipeline
            response_audio = await self.pipeline.process_audio(chunk, self.sample_rate)
            
            if response_audio and self.bridge:
                await self.bridge.send_audio(response_audio, self.sample_rate)
                logger.info(f"📤 Sent AI response to Meet: {len(response_audio)} bytes")
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
        
        finally:
            self.is_processing = False
    
    def has_speech(self, audio_array: np.ndarray) -> bool:
        """
        Simple speech detection based on audio energy.
        
        Args:
            audio_array: Audio data as numpy array
            
        Returns:
            True if likely contains speech
        """
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
        
        # Threshold for speech detection (adjust as needed)
        speech_threshold = 300
        
        has_speech = rms > speech_threshold
        
        if has_speech:
            logger.info(f"🎙️ Speech detected (RMS: {rms:.1f})")
        
        return has_speech

    def generate_test_audio(self, duration_ms: int = 500, frequency: int = 440) -> bytes:
        """
        Generate a test audio tone (for testing the audio loop).
        
        Args:
            duration_ms: Duration in milliseconds
            frequency: Frequency in Hz (440 = A4 note)
            
        Returns:
            PCM audio bytes
        """
        num_samples = int((duration_ms / 1000.0) * self.sample_rate)
        t = np.linspace(0, duration_ms / 1000.0, num_samples)
        
        # Generate sine wave
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope to avoid clicks
        envelope = np.ones_like(audio)
        fade_samples = int(0.01 * self.sample_rate)  # 10ms fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        audio *= envelope
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767 * 0.3).astype(np.int16)  # 30% volume
        
        return audio_int16.tobytes()

    async def start(self, meeting_url: str):
        """
        Start the voice agent and join Google Meet.
        
        Args:
            meeting_url: Google Meet URL to join
        """
        try:
            # Step 1: Start WebSocket server
            logger.info("=" * 60)
            logger.info("STEP 1: Starting WebSocket Server")
            logger.info("=" * 60)
            
            host = os.getenv("WEBSOCKET_SERVER_HOST", "0.0.0.0")
            port = int(os.getenv("WEBSOCKET_SERVER_PORT", "8765"))
            
            self.bridge = AttendeeBridge(
                host=host,
                port=port,
                audio_callback=self.audio_from_meet_callback
            )
            
            await self.bridge.start()
            
            logger.info(f"\n📍 Your WebSocket server is running on port {port}")
            logger.info(f"📍 Make sure ngrok is exposing this port!")
            logger.info(f"📍 Run: ngrok http {port}\n")
            
            # Step 2: Check ngrok URL
            logger.info("=" * 60)
            logger.info("STEP 2: Verifying ngrok URL")
            logger.info("=" * 60)
            
            self.meet_session = MeetSession()
            
            try:
                websocket_url = self.meet_session.get_websocket_url()
                logger.info(f"✓ WebSocket public URL: {websocket_url}\n")
            except ValueError as e:
                logger.error(f"\n❌ {e}")
                logger.error("\nPlease:")
                logger.error("1. Run: ngrok http 8765")
                logger.error("2. Copy the 'Forwarding' URL (e.g., https://abc123.ngrok.io)")
                logger.error("3. Update WEBSOCKET_PUBLIC_URL in .env.local")
                logger.error("4. Restart this script\n")
                return
            
            # Step 3: Create bot and join meeting
            logger.info("=" * 60)
            logger.info("STEP 3: Creating Bot and Joining Meeting")
            logger.info("=" * 60)
            
            bot_id = await self.meet_session.create_bot(
                meeting_url=meeting_url,
                websocket_url=websocket_url,
                bot_name="Jarvis"
            )
            
            logger.info(f"\n✓ Bot created: {bot_id}")
            logger.info(f"✓ Joining meeting: {meeting_url}\n")
            
            # Step 4: Monitor and run
            logger.info("=" * 60)
            logger.info("STEP 4: Voice Agent Running")
            logger.info("=" * 60)
            logger.info("\n🎤 The bot is now in the meeting!")
            logger.info("🔊 Try speaking in the meeting...")
            logger.info("📊 Watch the logs below for audio activity\n")
            logger.info("Press Ctrl+C to stop\n")
            
            # Start monitoring bot state in background
            monitor_task = asyncio.create_task(
                self.meet_session.monitor_bot_state(interval=10)
            )
            
            # Wait for monitoring to complete or user interrupt
            await monitor_task
            
        except KeyboardInterrupt:
            logger.info("\n\n⏹️  Shutting down...")
        except Exception as e:
            logger.error(f"\n❌ Error: {e}", exc_info=True)
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        
        if self.meet_session:
            try:
                await self.meet_session.leave_meeting()
            except Exception as e:
                logger.error(f"Error leaving meeting: {e}")
        
        if self.bridge:
            try:
                await self.bridge.stop()
            except Exception as e:
                logger.error(f"Error stopping bridge: {e}")
        
        logger.info("✓ Cleanup complete")


async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("\n" + "=" * 60)
        print("Google Meet Voice Agent - Usage")
        print("=" * 60)
        print("\nUsage:")
        print("  python meet_agent.py <google_meet_url>")
        print("\nExample:")
        print("  python meet_agent.py https://meet.google.com/abc-defg-hij")
        print("\nPrerequisites:")
        print("  1. Update .env.local with your credentials")
        print("  2. Run 'ngrok http 8765' in another terminal")
        print("  3. Copy ngrok URL to WEBSOCKET_PUBLIC_URL in .env.local")
        print("=" * 60 + "\n")
        sys.exit(1)
    
    meeting_url = sys.argv[1]
    
    agent = MeetVoiceAgent()
    await agent.start(meeting_url)


if __name__ == "__main__":
    asyncio.run(main())
