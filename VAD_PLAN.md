# Voice Agent VAD Implementation Plan

## Problem Statement

The voice agent was breaking long questions into chunks every 1.5s (hardcoded) and had jittery audio output. Need VAD-based speech detection with proper interruption handling.

## Architecture Design

### 1. Voice Activity Detection (Silero VAD)

- **Model**: LiveKit Silero VAD plugin (production-ready)
- **Configuration**:
  - Min speech duration: 200ms (filter out mouth sounds)
  - Min silence duration: 700ms (end of utterance detection)
  - Max utterance: 30s (safety limit)
  - Sample rate: 16kHz (matching Attendee.dev)

### 2. State Machine

```
AgentState enum:
- IDLE: Waiting for speech
- COLLECTING: Actively collecting speech audio
- PROCESSING: Transcribing + LLM + TTS
- BOT_SPEAKING: Sending audio to Meet
- INTERRUPTED: User spoke during bot response
```

### 3. Audio Pipeline Flow

**Input Flow (Google Meet → Agent)**:

```
Attendee.dev (20ms PCM chunks)
  → audio_from_meet_callback()
  → Convert to rtc.AudioFrame
  → VAD.push_frame()
  → _vad_processor_task() [async iterator]
  → _handle_vad_event()
```

**VAD Events**:

- `START_OF_SPEECH`: User started speaking
  - If state == IDLE: transition to COLLECTING
  - If state == BOT_SPEAKING: **INTERRUPTION** (clear output queue, transition to INTERRUPTED)
- `INFERENCE_DONE`: Continue collecting audio
  - Extract frame.data and append to audio_buffer
- `END_OF_SPEECH`: User stopped speaking
  - Transition to PROCESSING
  - Send complete utterance to streaming pipeline

**Output Flow (Agent → Google Meet)**:

```
_process_complete_utterance()
  → STT (OpenAI Whisper)
  → LLM (Gemini 2.0 Flash streaming)
  → TTS per sentence (Google Cloud TTS)
  → output_queue.put(audio_chunk)
  → _audio_sender_task() [background]
  → bridge.send_audio() [non-blocking with delays]
```

### 4. Non-Blocking Audio Streaming

- **Output Queue**: asyncio.Queue for audio chunks
- **Background Sender**: Separate task that consumes queue
- **No blocking**: STT/LLM/TTS pipeline never blocks on audio transmission
- **Chunk timing**: 95% of chunk duration delay to prevent WebSocket timeout

### 5. Interruption Handling

**Detection**:

- VAD detects START_OF_SPEECH while state == BOT_SPEAKING
- Increment interruptions counter
- Clear output queue immediately
- Transition to INTERRUPTED state

**Recovery**:

- Continue collecting new user speech
- Process new utterance normally
- Partial bot response is abandoned

### 6. Conversation History Management

**CRITICAL**: Maintain conversation context across turns

**Structure**:

```python
self.conversation_history = []  # List of message dicts

Format:
[
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hello there! How can I help?"},
  {"role": "user", "content": "What's the weather?"},
  {"role": "assistant", "content": "The temperature is..."}
]
```

**LLM Integration**:

- Pass FULL history to LLM on every request
- Let LLM decide what context to use (no forced context awareness)
- Append user utterance BEFORE LLM call
- Append assistant response AFTER TTS generation

**Not Hardcoded**: No system prompt forcing context usage - LLM naturally uses history

## Key Design Decisions

### Why Async Iterator for VAD?

LiveKit VADStream uses async iterator pattern, not event polling:

```python
async for event in self.vad_stream:
    # Process events
```

### Why Background Sender Task?

Prevents pipeline blocking:

- STT/LLM/TTS can continue processing
- Audio streams immediately as generated
- Interruptions can clear queue without waiting

### Why Frame Data Extraction?

VAD events contain AudioFrame objects, need to extract raw bytes:

```python
for frame in event.frames:
    self.audio_buffer.extend(frame.data)  # Extract .data property
```

### Why is_running Flag?

Prevents operations after cleanup:

- Stop accepting new audio immediately on Ctrl+C
- Avoid "input ended" errors during shutdown
- Clean graceful termination

## Production Requirements Met

✅ **No hardcoded time chunks** - VAD detects natural speech boundaries
✅ **Complete utterance capture** - 700ms silence detection ensures full sentences
✅ **Smooth audio streaming** - Non-blocking sender with proper timing
✅ **Interruption support** - User can interrupt bot responses
✅ **State machine** - Clear state transitions and error recovery
✅ **Metrics tracking** - speech_events, utterances_processed, interruptions, total_turns

## Known Issues to Address

1. **Conversation history not implemented** - Bot reintroduces itself every turn
2. **Interruption not triggering** - Need to verify state machine logic
3. **Bot cleanup fails** - Attendee.dev API state mismatch on termination

## Next Steps

1. Add conversation history to livekit_pipeline.py
2. Debug interruption detection (check BOT_SPEAKING state timing)
3. Fix bot removal API call (handle "joined_recording" state)
4. Consider optimizations:
   - Streaming TTS per sentence vs full response
   - Parallel STT + LLM for lower latency
   - Dynamic VAD sensitivity based on ambient noise
