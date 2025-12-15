"""
Main integration script for Google Meet voice agent.
Connects Attendee.dev WebSocket bridge with LiveKit voice agent.
"""

import asyncio
import logging
import os
import sys
from typing import Optional
from enum import Enum
from dotenv import load_dotenv
import numpy as np
from livekit.plugins import silero
from livekit import rtc

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


class AgentState(Enum):
    """Conversation state machine states."""
    IDLE = "idle"                    # No speech, waiting
    COLLECTING = "collecting"        # Speech detected, buffering
    PROCESSING = "processing"        # Utterance complete, running STT→LLM
    BOT_SPEAKING = "bot_speaking"    # Sending response to Meet
    INTERRUPTED = "interrupted"      # User spoke during bot response


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
        
        # Audio configuration
        self.sample_rate = 16000
        
        # LiveKit STT → LLM → TTS pipeline
        self.pipeline = LiveKitPipeline()
        
        # Silero VAD (initialized in start())
        self.vad_model: Optional[silero.VAD] = None
        self.vad_stream: Optional[silero.VADStream] = None
        
        # State machine
        self.state = AgentState.IDLE
        self.audio_buffer = bytearray()
        
        # Output queue for non-blocking audio sending
        self.output_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self.sender_task: Optional[asyncio.Task] = None
        self.vad_processor_task: Optional[asyncio.Task] = None
        self.is_running = False  # Flag to prevent operations after cleanup
        self.sender_active = False  # Track if audio sender is actively sending
        
        # Metrics
        self.metrics = {
            "speech_events": 0,
            "utterances_processed": 0,
            "interruptions": 0,
            "total_turns": 0
        }
        
        logger.info("MeetVoiceAgent initialized - ready for VAD setup")

    async def audio_from_meet_callback(self, pcm_data: bytes, sample_rate: int):
        """
        Callback when audio is received from Google Meet via Attendee.dev.
        Uses Silero VAD to detect complete utterances.
        
        Args:
            pcm_data: Raw PCM audio from meeting (16-bit, mono)
            sample_rate: Sample rate of the audio
        """
        if not self.vad_stream or not self.is_running:
            return
        
        # Convert raw PCM bytes to AudioFrame for VAD detection
        # pcm_data is 16-bit signed int, mono
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Create AudioFrame with correct format for VAD
        frame = rtc.AudioFrame(
            data=audio_array.tobytes(),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=len(audio_array)
        )
        
        # Feed audio frame to VAD for speech detection (non-blocking)
        self.vad_stream.push_frame(frame)
        
        # IMPORTANT: Store ORIGINAL raw PCM for transcription (not VAD-processed audio)
        # VAD resamples internally, we want the original quality for STT
        if self.state == AgentState.COLLECTING:
            self.audio_buffer.extend(pcm_data)
    
    async def _vad_processor_task(self):
        """
        Background task that processes VAD events from the async iterator.
        """
        logger.info("🎙️ VAD processor task started")
        
        try:
            async for event in self.vad_stream:
                logger.debug(f"VAD event received: {event.type}")
                await self._handle_vad_event(event)
        except asyncio.CancelledError:
            logger.info("VAD processor task cancelled")
        except Exception as e:
            logger.error(f"VAD processor task error: {e}", exc_info=True)

    async def _handle_vad_event(self, event):
        """
        Handle VAD events for speech detection.
        
        Args:
            event: VAD event from Silero
        """
        from livekit.agents import vad
        
        if event.type == vad.VADEventType.START_OF_SPEECH:
            await self._on_speech_start()
            
        elif event.type == vad.VADEventType.INFERENCE_DONE:
            # VAD inference completed - we don't need to collect frames here
            # Original audio is being collected in audio_from_meet_callback()
            pass
            
        elif event.type == vad.VADEventType.END_OF_SPEECH:
            await self._on_speech_end()
    
    async def _on_speech_start(self):
        """Handle start of speech event."""
        self.metrics["speech_events"] += 1
        
        if self.sender_active:
            # User interrupted the bot while it was actively sending audio
            logger.info("🛑 User interrupted bot response")
            self.metrics["interruptions"] += 1
            
            # Clear output queue
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            # Transition to collecting immediately
            self.state = AgentState.COLLECTING
            self.audio_buffer = bytearray()
            logger.info("🎤 Speech started - collecting audio (interrupted bot)")
            return
        
        # Start collecting speech from idle
        if self.state == AgentState.IDLE:
            logger.info("🎤 Speech started - collecting audio")
            self.state = AgentState.COLLECTING
            self.audio_buffer = bytearray()
    
    async def _on_speech_end(self):
        """Handle end of speech event - process complete utterance."""
        if self.state != AgentState.COLLECTING:
            return
        
        utterance_audio = bytes(self.audio_buffer)
        self.audio_buffer = bytearray()
        
        duration_sec = len(utterance_audio) / (self.sample_rate * 2)
        logger.info(f"🔇 Speech ended - collected {len(utterance_audio)} bytes ({duration_sec:.2f}s)")
        
        if len(utterance_audio) < 3200:  # Less than 0.1s of audio
            logger.debug("Utterance too short, ignoring")
            self.state = AgentState.IDLE
            return
        
        # Process complete utterance
        self.state = AgentState.PROCESSING
        await self._process_complete_utterance(utterance_audio)
    
    async def _process_complete_utterance(self, audio_pcm: bytes):
        """
        Process complete utterance through streaming STT → LLM → TTS pipeline.
        
        Args:
            audio_pcm: Complete utterance audio
        """
        try:
            self.metrics["utterances_processed"] += 1
            self.metrics["total_turns"] += 1
            
            logger.info("🔄 Processing complete utterance through streaming pipeline")
            
            # Use streaming pipeline
            self.state = AgentState.BOT_SPEAKING
            
            chunk_count = 0
            async for audio_chunk in self.pipeline.process_audio_streaming(audio_pcm, self.sample_rate):
                # Check for interruption (state changed by _on_speech_start)
                if self.state == AgentState.COLLECTING:
                    logger.info("⏹️ Stopping response due to interruption")
                    return  # Exit immediately, new speech is being collected
                
                # Queue audio chunk for sending (non-blocking)
                try:
                    await asyncio.wait_for(
                        self.output_queue.put(audio_chunk),
                        timeout=1.0
                    )
                    chunk_count += 1
                except asyncio.TimeoutError:
                    logger.warning("Output queue full, dropping chunk")
            
            logger.info(f"✅ Queued {chunk_count} audio chunks for streaming")
            
            # Return to idle after speaking (if not interrupted)
            if self.state == AgentState.BOT_SPEAKING:
                self.state = AgentState.IDLE
                
        except Exception as e:
            logger.error(f"Error processing utterance: {e}", exc_info=True)
            self.state = AgentState.IDLE
    
    async def _audio_sender_task(self):
        """
        Background task that continuously sends audio chunks to Meet.
        This prevents blocking the main VAD processing loop.
        """
        logger.info("🔊 Audio sender task started")
        
        try:
            while True:
                # Get next audio chunk from queue
                audio_chunk = await self.output_queue.get()
                
                if audio_chunk is None:  # Poison pill to stop
                    logger.info("Audio sender task stopping")
                    break
                
                # Mark sender as active (bot is speaking)
                self.sender_active = True
                
                # Send audio to Meet (this call chunks and delays internally)
                if self.bridge:
                    try:
                        await self.bridge.send_audio(audio_chunk, self.sample_rate)
                    except Exception as e:
                        logger.error(f"Error sending audio chunk: {e}")
                
                self.output_queue.task_done()
                
                # Mark inactive if queue is empty (bot finished speaking)
                if self.output_queue.empty():
                    self.sender_active = False
                
        except asyncio.CancelledError:
            logger.info("Audio sender task cancelled")
        except Exception as e:
            logger.error(f"Audio sender task error: {e}", exc_info=True)
        finally:
            self.sender_active = False
    
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
            # Step 0: Initialize Silero VAD
            logger.info("=" * 60)
            logger.info("STEP 0: Initializing Voice Activity Detection")
            logger.info("=" * 60)
            
            logger.info("Loading Silero VAD model...")
            self.vad_model = silero.VAD.load(
                min_speech_duration=0.1,       # 100ms to catch first words faster
                min_silence_duration=0.8,      # 800ms to handle natural pauses
                prefix_padding_duration=0.3,   # 300ms padding before
                max_buffered_speech=30.0,      # Max 30s utterance
                activation_threshold=0.4,      # Sensitive detection
                sample_rate=self.sample_rate,  # Match Attendee (16kHz)
                force_cpu=True                 # Reliable performance
            )
            
            self.vad_stream = self.vad_model.stream()
            
            logger.info("✓ Silero VAD initialized")
            logger.info("  - Min speech: 100ms (fast detection)")
            logger.info("  - Min silence: 800ms (natural pauses)")
            logger.info("  - Prefix padding: 300ms (captures first words)")
            logger.info("  - Max utterance: 30s")
            logger.info("  - Sample rate: 16kHz")
            logger.info("")
            
            # Mark as running before starting tasks
            self.is_running = True
            
            # Start background audio sender task
            self.sender_task = asyncio.create_task(self._audio_sender_task())
            logger.info("✓ Background audio sender started")
            
            # Start background VAD processor task
            self.vad_processor_task = asyncio.create_task(self._vad_processor_task())
            logger.info("✓ Background VAD processor started\n")
            
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
        
        # Stop accepting new audio immediately
        self.is_running = False
        await asyncio.sleep(0.1)  # Brief pause for in-flight operations
        
        # Stop VAD processor task
        if self.vad_processor_task:
            try:
                self.vad_processor_task.cancel()
                await asyncio.wait_for(self.vad_processor_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception as e:
                logger.error(f"Error stopping VAD processor: {e}")
        
        # Close VAD stream
        if self.vad_stream:
            try:
                await self.vad_stream.aclose()
            except Exception as e:
                logger.error(f"Error closing VAD stream: {e}")
                # Stop audio sender task
        if self.sender_task:
            try:
                await self.output_queue.put(None)  # Poison pill
                await asyncio.wait_for(self.sender_task, timeout=2.0)
            except asyncio.TimeoutError:
                self.sender_task.cancel()
            except Exception as e:
                logger.error(f"Error stopping sender task: {e}")
        
        # Print metrics
        logger.info("\n📊 Session Metrics:")
        for key, value in self.metrics.items():
            logger.info(f"  {key}: {value}")
        logger.info("")
        
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
