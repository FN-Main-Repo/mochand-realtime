# Google Meet Voice Agent Integration Plan

## 📋 Executive Summary

This document outlines the integration plan for connecting your LiveKit voice agent with Google Meet using **Attendee.dev** as the meeting bot bridge.

---

## 🔍 Current Architecture Analysis

### Your Existing Flow (LiveKit Native)

```
User Voice → LiveKit → STT (OpenAI) → agent.py (LLM: Gemini via OpenRouter) → TTS (Google) → LiveKit → Voice Output
```

### Your Current Stack:

| Component          | Technology                               |
| ------------------ | ---------------------------------------- |
| **STT**            | OpenAI STT                               |
| **LLM**            | Google Gemini 2.0 Flash (via OpenRouter) |
| **TTS**            | Google TTS                               |
| **VAD**            | Silero VAD                               |
| **Turn Detection** | Multilingual Model                       |
| **Framework**      | LiveKit Agents SDK v1.2                  |

---

## 🎯 Target Architecture (Google Meet Integration)

### Proposed Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GOOGLE MEET                                     │
│                                                                             │
│   User speaks in Meet  ─────►  Attendee Bot (joins as participant)          │
│                                      │                                       │
│   User hears bot response ◄─────────┘                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │ WebSocket (bidirectional)
                                       │ Raw PCM Audio Streaming
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         YOUR WEBSOCKET SERVER                                │
│                                                                             │
│   Receive: {"trigger": "realtime_audio.mixed", "data": {"chunk": "...", ... │
│   Send:    {"trigger": "realtime_audio.bot_output", "data": {"chunk": "..."}}│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │ WebSocket / Internal
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            LIVEKIT AGENTS                                    │
│                                                                             │
│   Raw Audio ──► STT (OpenAI) ──► LLM (Gemini) ──► TTS (Google) ──► Raw Audio│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ✅ Attendee.dev Capabilities (Confirmed)

Based on my research of the Attendee.dev documentation and GitHub repository:

### What Attendee.dev SUPPORTS for Your Use Case:

| Feature                     | Status                   | Details                              |
| --------------------------- | ------------------------ | ------------------------------------ |
| **Google Meet Bot**         | ✅ Supported             | Bot joins as a participant           |
| **Real-time Audio Input**   | ✅ Supported             | Receive raw audio via WebSocket      |
| **Real-time Audio Output**  | ✅ Supported             | Send raw audio via WebSocket         |
| **Bidirectional WebSocket** | ✅ Supported             | Single WebSocket for both directions |
| **Low Latency Streaming**   | ✅ Supported             | Designed for voice agent use cases   |
| **Sample Rates**            | ✅ 8kHz, 16kHz, 24kHz    | Configurable per bot                 |
| **PCM Audio Format**        | ✅ 16-bit Single Channel | Base64 encoded in JSON               |

### Attendee WebSocket Message Format:

**Receiving Audio FROM Meeting (User speaks):**

```json
{
  "bot_id": "bot_12345abcdef",
  "trigger": "realtime_audio.mixed",
  "data": {
    "chunk": "BASE64_ENCODED_PCM_AUDIO",
    "sample_rate": 16000,
    "timestamp_ms": 1703123456789
  }
}
```

**Sending Audio TO Meeting (Bot speaks):**

```json
{
  "trigger": "realtime_audio.bot_output",
  "data": {
    "chunk": "BASE64_ENCODED_PCM_AUDIO",
    "sample_rate": 16000
  }
}
```

---

## 🚨 Critical Architecture Decision

### Option A: LiveKit as Audio Pipeline (Recommended for Low Latency)

Use LiveKit's built-in STT/TTS pipeline with a custom audio transport layer.

**Pros:**

- Leverages your existing agent.py setup
- LiveKit's turn detection and VAD work seamlessly
- Lower latency with preemptive generation
- Noise cancellation built-in

**Cons:**

- Need to bridge WebSocket audio to LiveKit room
- More complex setup

### Option B: Direct Audio Processing (Simpler but More Work)

Process audio directly without LiveKit room, using individual STT/LLM/TTS calls.

**Pros:**

- Simpler architecture
- No LiveKit room needed

**Cons:**

- Lose LiveKit's turn detec2tion, VAD, noise cancellation
- Need to handle audio buffering yourself
- Higher latency potential

### **Recommendation: Option A (LiveKit as Audio Pipeline)**

---

## 📝 Implementation Plan

### Phase 1: Prerequisites & Setup

#### 1.1 Attendee.dev Account Setup

- [ ] Sign up at https://app.attendee.dev/accounts/signup/
- [ ] Obtain API key from the UI (Settings → API Keys)
- [ ] Note: No special OAuth for Google Meet (Attendee handles this via Chrome)

#### 1.2 Environment Variables Needed

```env
# Existing
OPENROUTER_API_KEY=your_key
GOOGLE_CREDENTIALS_FILE=path_to_credentials

# New - Attendee.dev
ATTENDEE_API_KEY=your_attendee_api_key
ATTENDEE_API_URL=https://app.attendee.dev/api/v1

# WebSocket Server
WEBSOCKET_SERVER_HOST=0.0.0.0
WEBSOCKET_SERVER_PORT=8765
WEBSOCKET_PUBLIC_URL=wss://your-domain.com/attendee-websocket
```

#### 1.3 Dependencies to Add

```toml
# Add to pyproject.toml
dependencies = [
    # ... existing deps
    "aiohttp>=3.9.0",          # For Attendee REST API
    "websockets>=12.0",        # For WebSocket server
]
```

---

### Phase 2: WebSocket Server Implementation

Create a WebSocket server that:

1. Attendee.dev connects to (receives meeting audio)
2. Forwards audio to LiveKit
3. Receives TTS audio from LiveKit
4. Sends back to Attendee.dev

#### Key Components:

```
src/
├── agent.py              # Existing - modify for custom audio source
├── assistant.py          # Existing
├── llm.py               # Existing
├── attendee_bridge.py   # NEW - WebSocket server + Attendee API client
├── audio_transport.py   # NEW - Audio format conversion utilities
└── meet_session.py      # NEW - Manages Meet session lifecycle
```

---

### Phase 3: Implementation Details

#### 3.1 Attendee Bot Creation Flow

```python
# Create bot via REST API
POST https://app.attendee.dev/api/v1/bots
{
    "meeting_url": "https://meet.google.com/xxx-yyyy-zzz",
    "bot_name": "Mochand Assistant",
    "websocket_settings": {
        "audio": {
            "url": "wss://your-server.com/attendee-websocket",
            "sample_rate": 16000
        }
    }
}
```

#### 3.2 Audio Format Specifications

| Parameter   | Value                       |
| ----------- | --------------------------- |
| Format      | PCM (Pulse Code Modulation) |
| Bit Depth   | 16-bit                      |
| Channels    | 1 (Mono)                    |
| Sample Rate | 16000 Hz (recommended)      |
| Encoding    | Base64                      |

#### 3.3 LiveKit Integration Approach

**Challenge:** LiveKit expects to connect to a room with real-time media tracks, but Attendee.dev uses WebSocket for audio.

**Solution:** Create a "virtual participant" in LiveKit that:

1. Receives audio from Attendee WebSocket
2. Publishes as audio track to LiveKit room
3. Subscribes to agent's audio response
4. Sends back via Attendee WebSocket

---

### Phase 4: Latency Optimization

#### Expected Latency Breakdown:

| Component                 | Estimated Latency |
| ------------------------- | ----------------- |
| Meet → Attendee Bot       | ~50-100ms         |
| Attendee → Your WebSocket | ~50-100ms         |
| STT Processing            | ~200-500ms        |
| LLM Response (streaming)  | ~300-1000ms       |
| TTS Processing            | ~200-500ms        |
| Your WebSocket → Attendee | ~50-100ms         |
| Attendee Bot → Meet       | ~50-100ms         |
| **Total End-to-End**      | **~900ms - 2.5s** |

#### Optimization Strategies:

1. **Use streaming STT** - Get partial transcripts
2. **Use streaming LLM** - Already using with OpenRouter
3. **Use streaming TTS** - Google TTS supports this
4. **Preemptive generation** - Already enabled in your agent.py
5. **Low sample rate** - 16kHz is good balance
6. **Chunked audio sending** - Send small chunks frequently

---

## 🔧 What I Need From You

### Required Information:

1. **Domain/Server for WebSocket:**

   - Do you have a public domain/IP where the WebSocket server can be hosted?
   - Will you use ngrok for testing, or do you have cloud infrastructure?

2. **Attendee.dev Account:**

   - Have you signed up for Attendee.dev?
   - Do you have an API key?

3. **Deployment Environment:**

   - Where will this run? (AWS, GCP, Azure, local, etc.)
   - Do you need Docker support?

4. **Testing Approach:**
   - Do you have a Google Meet account for testing?
   - Any specific meeting scenarios to support?

### Optional Clarifications:

1. Do you want the bot to:
   - Join automatically when meeting URL is provided via API?
   - Have a specific avatar/name?
   - Record the conversation?
2. Multi-language support:
   - Your current setup supports multiple Indian languages
   - Should this work in Google Meet too?

---

## 📊 Alternative Approaches Considered

### 1. Recall.ai

- Similar to Attendee.dev
- Commercial product, more expensive
- More mature but closed-source

### 2. Direct Google Meet API

- Google doesn't provide SDK for bots
- Would need to run Chrome in headless mode yourself
- Much more complex

### 3. LiveKit Meet

- LiveKit's own video conferencing
- Won't work with Google Meet
- Only for LiveKit-native meetings

**Verdict:** Attendee.dev is the best choice for your use case - it's open-source, supports real-time audio streaming, and specifically designed for voice agent integration.

---

## 🔗 Useful References

1. **Attendee.dev Realtime Audio Docs:** https://github.com/attendee-labs/attendee/blob/main/docs/realtime_audio.md
2. **Voice Agent Example:** https://github.com/attendee-labs/voice-agent-example
3. **Attendee API Reference:** https://docs.attendee.dev/
4. **LiveKit Agents Docs:** https://docs.livekit.io/agents/

---

## ⏭️ Next Steps

Once you provide the required information, I will:

1. **Create `src/attendee_bridge.py`** - WebSocket server + Attendee API client
2. **Create `src/audio_transport.py`** - Audio conversion utilities
3. **Create `src/meet_session.py`** - Session lifecycle management
4. **Modify `src/agent.py`** - Add custom audio source support
5. **Update `pyproject.toml`** - Add new dependencies
6. **Create test scripts** - For end-to-end testing
7. **Create `docker-compose.yml`** - For easy deployment

---

## ⚠️ Important Notes

1. **Attendee.dev's WebSocket Connection:**

   - Attendee CONNECTS TO your WebSocket server (you host the server)
   - It retries up to 30 times with 2-second delays
   - Your server must be publicly accessible via `wss://`

2. **Audio Chunk Size:**

   - Attendee sends audio in chunks (typically 100ms each)
   - You should send responses in similar chunk sizes

3. **Bot Lifecycle:**

   - Bots go through states: Ready → Joining → Joined → Leaving → Ended
   - Handle webhooks for state changes if needed

4. **Rate Limits:**
   - Check Attendee.dev's rate limits for your plan
   - Free tier might have limitations

---

**Please review this plan and provide the requested information so I can proceed with implementation!**
