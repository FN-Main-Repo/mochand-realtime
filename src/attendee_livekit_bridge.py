import asyncio
import base64
import json
import logging
import websockets
from typing import Optional, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("attendee-bridge")

# Silence websockets debug spam
logging.getLogger("websockets").setLevel(logging.WARNING)


class AttendeeLiveKitBridge:
    """Manages audio routing between Attendee.dev and LiveKit"""
    
    def __init__(self, room_name: str, sample_rate: int = 16000):
        self.room_name = room_name
        self.sample_rate = sample_rate
        self.ws: Optional[websockets.WebSocketServerProtocol] = None
        self.running = False
        
        # Queues for bidirectional audio
        self.incoming_audio = asyncio.Queue()  # From Attendee -> LiveKit
        self.outgoing_audio = asyncio.Queue()  # From LiveKit -> Attendee
        
        # Stats
        self.chunks_received = 0
        self.chunks_sent = 0
        
    async def handle_websocket(self, websocket):
        """Handle Attendee.dev WebSocket connection"""
        logger.info(f"🔗 Attendee bot connected for room: {self.room_name}")
        self.ws = websocket
        self.running = True
        
        # Start sender task
        sender_task = asyncio.create_task(self._send_audio_loop())
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get("trigger") == "realtime_audio.mixed":
                        pcm_b64 = data["data"]["chunk"]
                        pcm_bytes = base64.b64decode(pcm_b64)
                        await self.incoming_audio.put(pcm_bytes)
                        
                        self.chunks_received += 1
                        if self.chunks_received % 50 == 0:
                            logger.info(f"🎤 Received {self.chunks_received} chunks from Meet")
                    
                    elif data.get("trigger") == "bot.joined":
                        logger.info("✅ Bot joined Google Meet")
                    
                    elif data.get("trigger") == "bot.left":
                        logger.warning("⚠️ Bot left Google Meet")
                        break
                        
                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 WebSocket closed")
        finally:
            self.running = False
            sender_task.cancel()
            logger.info(f"📊 Session stats - RX: {self.chunks_received}, TX: {self.chunks_sent}")
            
    def clear_outgoing_audio(self):
        """
        Immediately drop any queued outbound audio.
        Used for barge-in / interruption.
        """
        try:
            while not self.outgoing_audio.empty():
                self.outgoing_audio.get_nowait()
        except Exception:
            pass

    
    async def _send_audio_loop(self):
        """Continuously send queued audio to Attendee.dev"""
        logger.info("🔊 Started outgoing audio loop")
        while self.running and self.ws:
            try:
                pcm_bytes = await asyncio.wait_for(self.outgoing_audio.get(), timeout=0.1)
                
                # Correct Attendee.dev format
                payload = {
                    "trigger": "realtime_audio.bot_output",
                    "data": {
                        "chunk": base64.b64encode(pcm_bytes).decode('utf-8'),
                        "sample_rate": self.sample_rate
                    }
                }
                await self.ws.send(json.dumps(payload))
                
                self.chunks_sent += 1
                if self.chunks_sent % 50 == 0:
                    logger.info(f"🔊 Sent {self.chunks_sent} chunks to Meet")
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.error(f"❌ Error sending audio: {e}")
                break
        logger.info("🔊 Stopped outgoing audio loop")
    
    async def get_incoming_audio(self):
        """Get audio from Attendee (blocking)"""
        return await self.incoming_audio.get()
    
    async def send_audio(self, pcm_bytes: bytes):
        """Queue audio to send to Attendee"""
        if self.running:
            await self.outgoing_audio.put(pcm_bytes)


# Global default bridge
_default_bridge: Optional[AttendeeLiveKitBridge] = None


def get_or_create_default_bridge() -> AttendeeLiveKitBridge:
    """Get or create the default bridge"""
    global _default_bridge
    if _default_bridge is None:
        _default_bridge = AttendeeLiveKitBridge("default")
    return _default_bridge


def get_default_bridge() -> Optional[AttendeeLiveKitBridge]:
    """Get the default bridge (may be None if not created)"""
    return _default_bridge


class AttendeeServer:
    """WebSocket server for Attendee.dev bots"""
    
    def __init__(self):
        self.server = None
        
    async def start(self, host="0.0.0.0", port=8765):
        """Start WebSocket server"""
        logger.info(f"🚀 Starting WebSocket server on {host}:{port}")
        
        async def websocket_handler(websocket):
            """Handle new WebSocket connection - websockets v15+ API"""
            # Get path from request (v15+)
            path = websocket.request.path if hasattr(websocket, 'request') else "/"
            logger.info(f"📥 New connection from {websocket.remote_address} path={path}")
            
            # Always use the default shared bridge
            bridge = get_or_create_default_bridge()
            logger.info(f"📦 Using default bridge (running={bridge.running})")
            
            await bridge.handle_websocket(websocket)
        
        self.server = await websockets.serve(websocket_handler, host, port)
        logger.info(f"✅ WebSocket server listening on {host}:{port}")


# Global server instance
_server = AttendeeServer()


async def start_server(host="0.0.0.0", port=8765):
    """Start the global Attendee server"""
    # Create the bridge immediately so agent can find it
    get_or_create_default_bridge()
    logger.info("📦 Default bridge created and ready")
    
    await _server.start(host, port)