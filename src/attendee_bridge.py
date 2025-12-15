"""
Attendee.dev Bridge - WebSocket server and API client for Google Meet integration.
This module handles bidirectional audio streaming with Attendee.dev bots.
"""

import asyncio
import json
import logging
import os
import websockets
from websockets.server import WebSocketServerProtocol
from typing import Optional, Callable
from audio_transport import AudioTransport

logger = logging.getLogger("attendee-bridge")
logger.setLevel(logging.INFO)


class AttendeeBridge:
    """
    WebSocket server that bridges Attendee.dev bots with LiveKit agents.
    
    Flow:
    1. Attendee bot connects to this WebSocket server
    2. Receives audio from Google Meet via WebSocket
    3. Forwards audio to audio_callback for processing
    4. Receives processed audio via send_audio()
    5. Sends audio back to Attendee bot
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        audio_callback: Optional[Callable[[bytes, int], None]] = None
    ):
        """
        Initialize the Attendee bridge.
        
        Args:
            host: WebSocket server host
            port: WebSocket server port
            audio_callback: Callback function for incoming audio (pcm_data, sample_rate)
        """
        self.host = host
        self.port = port
        self.audio_callback = audio_callback
        
        self.websocket: Optional[WebSocketServerProtocol] = None
        self.bot_id: Optional[str] = None
        self.connected = False
        self.server = None
        
        # Audio configuration
        self.sample_rate = 16000  # Default sample rate
        
        # Statistics
        self.audio_chunks_received = 0
        self.audio_chunks_sent = 0

    async def start(self):
        """Start the WebSocket server."""
        logger.info(f"Starting Attendee bridge on {self.host}:{self.port}")
        
        self.server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info(f"✓ WebSocket server started on ws://{self.host}:{self.port}")
        logger.info(f"✓ Waiting for Attendee bot to connect...")

    async def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")

    async def _handle_connection(self, websocket: WebSocketServerProtocol):
        """
        Handle incoming WebSocket connections from Attendee.dev.
        
        Args:
            websocket: WebSocket connection
        """
        logger.info(f"New connection from {websocket.remote_address}")
        
        # Close any existing connection before accepting new one
        if self.websocket and self.connected:
            logger.info("Closing existing connection before accepting new one")
            try:
                await self.websocket.close()
            except:
                pass
        
        # Store the websocket connection
        self.websocket = websocket
        self.connected = True
        
        try:
            async for message in websocket:
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Attendee bot disconnected - waiting for reconnection...")
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
        finally:
            # Only mark as disconnected if this was the active connection
            if self.websocket == websocket:
                self.connected = False
                self.websocket = None
            logger.info("Connection closed")

    async def _handle_message(self, message: str):
        """
        Handle incoming messages from Attendee.dev.
        
        Args:
            message: JSON message string
        """
        try:
            data = json.loads(message)
            
            # Log first few messages for debugging
            if self.audio_chunks_received < 5:
                logger.info(f"Received message: {data.get('trigger', 'unknown')}")
            
            # Extract bot_id from first message
            if not self.bot_id and "bot_id" in data:
                self.bot_id = data["bot_id"]
                logger.info(f"✓ Connected to bot: {self.bot_id}")
            
            # Parse audio message
            result = AudioTransport.parse_attendee_audio_message(data)
            if result:
                pcm_data, sample_rate, timestamp_ms = result
                self.audio_chunks_received += 1
                
                # Log periodically
                if self.audio_chunks_received % 100 == 0:
                    logger.info(
                        f"📊 Stats: Received {self.audio_chunks_received} chunks, "
                        f"Sent {self.audio_chunks_sent} chunks"
                    )
                
                # Forward to audio callback
                if self.audio_callback:
                    await self.audio_callback(pcm_data, sample_rate)
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON message received")
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    async def send_audio(self, pcm_data: bytes, sample_rate: int = 16000, chunk_duration_ms: int = 20):
        """
        Send audio data to Attendee bot in small chunks for smooth playback.
        
        Args:
            pcm_data: Raw PCM audio bytes (16-bit, mono)
            sample_rate: Sample rate in Hz
            chunk_duration_ms: Duration of each chunk in milliseconds (default 20ms)
        """
        if not self.connected or not self.websocket:
            logger.warning("Cannot send audio - not connected to Attendee bot")
            return
        
        try:
            # Calculate chunk size in bytes (16-bit = 2 bytes per sample)
            chunk_size = int((chunk_duration_ms / 1000.0) * sample_rate * 2)
            
            # Split audio into chunks and send
            total_bytes = len(pcm_data)
            chunks_sent = 0
            
            for i in range(0, total_bytes, chunk_size):
                chunk = pcm_data[i:i + chunk_size]
                
                # Create Attendee message for this chunk
                message = AudioTransport.create_attendee_audio_message(chunk, sample_rate)
                
                # Send via WebSocket
                await self.websocket.send(json.dumps(message))
                
                chunks_sent += 1
                self.audio_chunks_sent += 1
                
                # Small delay to simulate real-time playback
                # Using 95% of chunk duration for smoother streaming
                await asyncio.sleep(chunk_duration_ms / 1000.0 * 0.95)
            
            duration_sec = total_bytes / (sample_rate * 2)
            logger.debug(f"✅ Sent {total_bytes} bytes in {chunks_sent} chunks ({duration_sec:.1f}s)")
            
        except Exception as e:
            logger.error(f"Error sending audio: {e}", exc_info=False)

    def is_connected(self) -> bool:
        """Check if connected to an Attendee bot."""
        return self.connected

    def get_stats(self) -> dict:
        """Get bridge statistics."""
        return {
            "connected": self.connected,
            "bot_id": self.bot_id,
            "audio_chunks_received": self.audio_chunks_received,
            "audio_chunks_sent": self.audio_chunks_sent
        }


async def run_bridge(
    host: str,
    port: int,
    audio_callback: Optional[Callable[[bytes, int], None]] = None
):
    """
    Run the Attendee bridge server.
    
    Args:
        host: WebSocket server host
        port: WebSocket server port
        audio_callback: Callback for incoming audio
    """
    bridge = AttendeeBridge(host, port, audio_callback)
    await bridge.start()
    
    # Keep running
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await bridge.stop()


if __name__ == "__main__":
    # Test mode
    async def test_callback(pcm_data: bytes, sample_rate: int):
        duration_ms = AudioTransport.get_audio_duration_ms(pcm_data, sample_rate)
        logger.info(f"Received {len(pcm_data)} bytes ({duration_ms:.1f}ms) at {sample_rate}Hz")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    host = os.getenv("WEBSOCKET_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("WEBSOCKET_SERVER_PORT", "8765"))
    
    asyncio.run(run_bridge(host, port, test_callback))
