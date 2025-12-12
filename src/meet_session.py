"""
Google Meet session management via Attendee.dev API.
Handles bot creation, lifecycle management, and monitoring.
"""

import aiohttp
import logging
import os
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv(".env.local")

logger = logging.getLogger("meet-session")
logger.setLevel(logging.INFO)


class MeetSession:
    """
    Manages Google Meet session via Attendee.dev API.
    
    Handles:
    - Bot creation and joining meetings
    - Bot state monitoring
    - Bot lifecycle management
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None
    ):
        """
        Initialize Meet session manager.
        
        Args:
            api_key: Attendee.dev API key (or from env)
            api_url: Attendee.dev API URL (or from env)
        """
        self.api_key = api_key or os.getenv("ATTENDEE_API_KEY")
        self.api_url = api_url or os.getenv("ATTENDEE_API_URL", "https://app.attendee.dev/api/v1")
        
        if not self.api_key:
            raise ValueError("ATTENDEE_API_KEY not found in environment or parameters")
        
        self.bot_id: Optional[str] = None
        self.bot_state: Optional[str] = None
        self.meeting_url: Optional[str] = None
        
        logger.info(f"MeetSession initialized with API URL: {self.api_url}")

    async def create_bot(
        self,
        meeting_url: str,
        websocket_url: str,
        bot_name: str = "Mochand Assistant",
        sample_rate: int = 16000
    ) -> str:
        """
        Create and join a bot to a Google Meet meeting.
        
        Args:
            meeting_url: Google Meet URL to join
            websocket_url: Your WebSocket server URL (wss://)
            bot_name: Display name for the bot
            sample_rate: Audio sample rate (8000, 16000, or 24000)
            
        Returns:
            Bot ID
        """
        logger.info(f"Creating bot for meeting: {meeting_url}")
        logger.info(f"WebSocket URL: {websocket_url}")
        logger.info(f"Sample rate: {sample_rate}Hz")
        
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            "websocket_settings": {
                "audio": {
                    "url": websocket_url,
                    "sample_rate": sample_rate
                }
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/bots",
                headers=headers,
                json=payload
            ) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    raise Exception(f"Failed to create bot: {response.status} - {error_text}")
                
                data = await response.json()
                self.bot_id = data.get("id")
                self.bot_state = data.get("state")
                self.meeting_url = meeting_url
                
                logger.info(f"✓ Bot created: {self.bot_id}")
                logger.info(f"✓ Initial state: {self.bot_state}")
                
                return self.bot_id

    async def get_bot_status(self) -> Dict:
        """
        Get current bot status.
        
        Returns:
            Dictionary with bot status information
        """
        if not self.bot_id:
            raise ValueError("No bot created yet")
        
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.api_url}/bots/{self.bot_id}",
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to get bot status: {response.status} - {error_text}")
                
                data = await response.json()
                self.bot_state = data.get("state")
                
                return data

    async def leave_meeting(self) -> bool:
        """
        Make the bot leave the meeting.
        
        Returns:
            True if successful
        """
        if not self.bot_id:
            raise ValueError("No bot created yet")
        
        logger.info(f"Making bot leave meeting: {self.bot_id}")
        
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{self.api_url}/bots/{self.bot_id}",
                headers=headers
            ) as response:
                if response.status not in [200, 204]:
                    error_text = await response.text()
                    logger.error(f"Failed to leave meeting: {response.status} - {error_text}")
                    return False
                
                logger.info("✓ Bot left meeting successfully")
                return True

    async def monitor_bot_state(self, interval: int = 5) -> None:
        """
        Monitor bot state periodically.
        
        Args:
            interval: Polling interval in seconds
        """
        import asyncio
        
        logger.info(f"Starting bot state monitoring (interval: {interval}s)")
        
        try:
            while True:
                try:
                    status = await self.get_bot_status()
                    state = status.get("state")
                    
                    if state != self.bot_state:
                        logger.info(f"Bot state changed: {self.bot_state} -> {state}")
                        self.bot_state = state
                    
                    # Log important states
                    if state in ["joining", "joined", "ended", "fatal_error"]:
                        logger.info(f"Current bot state: {state}")
                    
                    # Stop monitoring if bot ended or errored
                    if state in ["ended", "fatal_error"]:
                        logger.info(f"Bot session ended with state: {state}")
                        break
                
                except Exception as e:
                    logger.error(f"Error monitoring bot state: {e}")
                
                await asyncio.sleep(interval)
        
        except asyncio.CancelledError:
            logger.info("Bot state monitoring cancelled")

    def get_websocket_url(self) -> str:
        """
        Get the WebSocket URL that should be used for this session.
        
        Returns:
            WebSocket URL (wss://)
        """
        public_url = os.getenv("WEBSOCKET_PUBLIC_URL")
        if not public_url:
            raise ValueError(
                "WEBSOCKET_PUBLIC_URL not set in .env.local\n"
                "Please set it to your ngrok URL (e.g., wss://abc123.ngrok.io)"
            )
        
        # Ensure it's wss://
        if not public_url.startswith("wss://"):
            if public_url.startswith("ws://"):
                public_url = public_url.replace("ws://", "wss://", 1)
            elif public_url.startswith("http://"):
                public_url = public_url.replace("http://", "wss://", 1)
            elif public_url.startswith("https://"):
                public_url = public_url.replace("https://", "wss://", 1)
            else:
                public_url = f"wss://{public_url}"
        
        return public_url


async def test_meet_session():
    """Test the MeetSession class."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python meet_session.py <google_meet_url>")
        print("Example: python meet_session.py https://meet.google.com/abc-defg-hij")
        return
    
    meeting_url = sys.argv[1]
    
    session = MeetSession()
    
    try:
        # Get WebSocket URL
        websocket_url = session.get_websocket_url()
        
        # Create bot
        bot_id = await session.create_bot(
            meeting_url=meeting_url,
            websocket_url=websocket_url,
            bot_name="Test Bot"
        )
        
        print(f"\n✓ Bot created: {bot_id}")
        print(f"✓ Meeting: {meeting_url}")
        print(f"✓ WebSocket: {websocket_url}")
        print("\nMonitoring bot state (Ctrl+C to stop)...")
        
        # Monitor state
        await session.monitor_bot_state()
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        await session.leave_meeting()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(test_meet_session())
