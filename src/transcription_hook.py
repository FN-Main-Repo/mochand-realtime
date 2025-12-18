from fastapi import FastAPI, Request, Header, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import Optional, Any
import hmac
import hashlib
import base64
import json
import httpx
from dotenv import load_dotenv
load_dotenv('.env.local')
import os

app = FastAPI()

WEBHOOK_SECRET = os.getenv("ATTENDEE_SECRET_KEY")
ATTENDEE_API_KEY = os.getenv("ATTENDEE_API_KEY")
ATTENDEE_BASE_URL = "https://app.attendee.dev/api/v1"

# Pydantic models for transcript webhook payload
class TranscriptionData(BaseModel):
    transcript: str
    words: Optional[list] = None

class TranscriptUpdateData(BaseModel):
    speaker_name: str
    speaker_uuid: str
    speaker_user_uuid: Optional[str] = None
    speaker_is_host: bool
    timestamp_ms: int
    duration_ms: int
    transcription: TranscriptionData

class WebhookPayload(BaseModel):
    idempotency_key: str
    bot_id: str
    bot_metadata: Optional[Any] = None
    trigger: str
    data: dict


def launch_bot(meeting_url: str, webhook_url: str, bot_name: str = "AI Assistant") -> dict:
    """Launch an Attendee bot for the given meeting URL"""
    try:
        headers = {
            "Authorization": f"Token {ATTENDEE_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            
            "webhooks": [
                {
                    "url": webhook_url,
                    "triggers": [
                        "bot.state_change",
                        "transcript.update"
                    ]
                }
            ],
            

        }

        response = httpx.post(
            f"{ATTENDEE_BASE_URL}/bots",
            headers=headers,
            json=payload,
            timeout=30.0
        )

        
        if response.status_code >= 400:
            print("❌ Attendee API response body:", response.text)

        response.raise_for_status()
        result = response.json()

        print(f"✅ Bot created with ID: {result.get('id')}")
        print(f"   State: {result.get('state')}")
        print(f"   Meeting URL: {meeting_url}")

        return result

    except Exception as e:
        print(f"❌ Error launching bot: {e}")
        raise



def verify_signature(payload: dict, signature: str, secret: str) -> bool:
    """Verify the webhook signature from Attendee"""
    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    secret_decoded = base64.b64decode(secret)
    expected_signature = hmac.new(
        secret_decoded, 
        payload_json.encode("utf-8"), 
        hashlib.sha256
    ).digest()
    return hmac.compare_digest(base64.b64encode(expected_signature).decode("utf-8"), signature)


@app.post('/receive-transcription')
async def receive_transcription(
    request: Request,
    x_webhook_signature: Optional[str] = Header(None)
):
    payload = await request.json()
    
    # Verify signature
    if x_webhook_signature and WEBHOOK_SECRET:
        if not verify_signature(payload, x_webhook_signature, WEBHOOK_SECRET):
            raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Parse the webhook
    webhook = WebhookPayload(**payload)
    
    if webhook.trigger == "transcript.update":
        transcript_data = TranscriptUpdateData(**webhook.data)
        
        print(f"[{transcript_data.speaker_name}]: {transcript_data.transcription.transcript}")
        
        return {
            "status": "received",
            "bot_id": webhook.bot_id,
            "speaker": transcript_data.speaker_name,
            "text": transcript_data.transcription.transcript
        }
    
    if webhook.trigger == "bot.state_change":
        print(f"🤖 Bot state: {webhook.data.get('old_state')} → {webhook.data.get('new_state')}")
        return {"status": "received", "trigger": webhook.trigger}
    
    if webhook.trigger == "participant_events.join_leave":
        event_type = webhook.data.get("event_type")
        participant = webhook.data.get("participant_name")
        print(f"👤 Participant {event_type}: {participant}")
        return {"status": "received", "trigger": webhook.trigger}
    
    return {"status": "ignored", "trigger": webhook.trigger}


if __name__ == "__main__":
    import sys
    
    
    meeting_url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("MEETING_URL")
    webhook_url = os.getenv("WEBHOOK_URL", "https://your-ngrok-url.ngrok.io/receive-transcription")
    bot_name = os.getenv("BOT_NAME", "AI Assistant")
    
    if meeting_url:
        print("\n🚀 Launching bot before starting server...")
        try:
            launch_bot(meeting_url, webhook_url, bot_name)
        except Exception as e:
            print(f"❌ Failed to launch bot: {e}")
    else:
        print("⚠️  No meeting URL provided. Server will start without launching a bot.")
        print("   Pass meeting URL as argument or set MEETING_URL env var")
    
    print("\n🌐 Starting webhook server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)