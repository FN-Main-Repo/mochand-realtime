# Google Meet Voice Agent - Testing Guide

## 🚀 Quick Start

### Step 1: Install ngrok (if not installed)

Download from: https://ngrok.com/download

Or use Chocolatey:

```powershell
choco install ngrok
```

### Step 2: Start ngrok

Open a **NEW terminal window** and run:

```powershell
ngrok http 8765
```

You should see output like:

```
Forwarding    https://abc123.ngrok.io -> http://localhost:8765
```

**Copy the https URL** (e.g., `https://abc123.ngrok.io`)

### Step 3: Update .env.local

Open `.env.local` and set the `WEBSOCKET_PUBLIC_URL`:

```env
WEBSOCKET_PUBLIC_URL=wss://abc123.ngrok.io
```

**Note:** Change `https://` to `wss://` in the URL!

### Step 4: Run the Voice Agent

In your main terminal (with venv activated):

```powershell
# Make sure venv is activated
.\venv\Scripts\activate.ps1

# Run the agent with a Google Meet URL
python src\meet_agent.py https://meet.google.com/your-meeting-code
```

### Step 5: Watch the Logs

You should see:

```
============================================================
STEP 1: Starting WebSocket Server
============================================================
✓ WebSocket server started on ws://0.0.0.0:8765
✓ Waiting for Attendee bot to connect...

============================================================
STEP 2: Verifying ngrok URL
============================================================
✓ WebSocket public URL: wss://abc123.ngrok.io

============================================================
STEP 3: Creating Bot and Joining Meeting
============================================================
✓ Bot created: bot_xxxxx
✓ Joining meeting: https://meet.google.com/...

============================================================
STEP 4: Voice Agent Running
============================================================
🎤 The bot is now in the meeting!
🔊 Try speaking in the meeting...
📊 Watch the logs below for audio activity
```

### Step 6: Test the Audio Loop

1. **Join the Google Meet** meeting yourself
2. **Speak into your microphone**
3. Watch the logs - you should see:

   ```
   📥 Received audio from Meet: XXX bytes, XX.Xms @ 16000Hz
   🔄 Processing XXX bytes of audio...
   📤 Sent test audio back to Meet: XXX bytes
   ```

4. **Listen** - you should hear a test beep from the bot (this confirms the audio loop works!)

---

## 🧪 Testing Commands

### Test WebSocket Server Only

```powershell
python src\attendee_bridge.py
```

### Test Attendee API Only

```powershell
python src\meet_session.py https://meet.google.com/your-meeting-code
```

### Test Full Integration

```powershell
python src\meet_agent.py https://meet.google.com/your-meeting-code
```

---

## 📊 What You Should See

### Terminal Logs

```
2025-12-12 10:00:00 - meet-agent - INFO - 📥 Received audio from Meet: 3200 bytes, 100.0ms @ 16000Hz
2025-12-12 10:00:00 - meet-agent - INFO - 🔄 Processing 32000 bytes of audio...
2025-12-12 10:00:00 - meet-agent - INFO - 📤 Sent test audio back to Meet: 22050 bytes
2025-12-12 10:00:00 - attendee-bridge - INFO - 📊 Stats: Received 100 chunks, Sent 10 chunks
```

### Google Meet

- You'll see "Mochand Assistant" join the meeting
- When you speak, the bot will respond with a test beep (440Hz tone)

---

## ⚙️ Configuration

### Audio Settings (.env.local)

```env
WEBSOCKET_SERVER_PORT=8765          # Port for WebSocket server
WEBSOCKET_PUBLIC_URL=wss://your-ngrok-url.ngrok.io  # Your ngrok URL
```

### Sample Rates

- Default: 16000 Hz (recommended)
- Available: 8000, 16000, 24000 Hz
- Change in `meet_agent.py` if needed

---

## 🐛 Troubleshooting

### Issue: "WEBSOCKET_PUBLIC_URL not set"

**Solution:** Update `.env.local` with your ngrok URL

### Issue: Bot doesn't join meeting

**Solution:**

1. Check your Attendee.dev API key in `.env.local`
2. Verify the meeting URL is correct
3. Check terminal logs for errors

### Issue: No audio received

**Solution:**

1. Verify ngrok is running and URL is correct
2. Check WebSocket server logs
3. Speak clearly into your mic in the meeting

### Issue: Bot joins but doesn't respond

**Solution:**

1. Check the logs for "📥 Received audio"
2. Verify audio processing in logs
3. Make sure your speakers are on to hear the test beep

---

## 📈 Next Steps

Once you confirm the audio loop works (you hear the test beep), you can integrate with LiveKit's STT/LLM/TTS pipeline:

1. The current setup proves the bidirectional audio flow works
2. Next phase: Replace `generate_test_audio()` with actual LiveKit agent processing
3. Integration point is in `meet_agent.py` → `process_audio_chunk()` method

---

## 🔄 Complete Workflow

```
1. You speak in Google Meet
   ↓
2. Attendee bot captures audio
   ↓
3. Sent via WebSocket to your server (port 8765)
   ↓
4. attendee_bridge.py receives it
   ↓
5. meet_agent.py processes it
   ↓
6. (Currently) Generates test beep
   ↓
7. (Future) LiveKit STT → LLM → TTS
   ↓
8. Sent back via WebSocket
   ↓
9. Attendee bot speaks in meeting
   ↓
10. You hear the response!
```

---

## ℹ️ Important Notes

- **ngrok URL changes** every time you restart ngrok (free plan)
- Update `.env.local` each time you get a new ngrok URL
- Port 8765 must be free on your machine
- Bot will stay in meeting until you press Ctrl+C or meeting ends

---

**Ready to test? Start with Step 1! 🚀**
