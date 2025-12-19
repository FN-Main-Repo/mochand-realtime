# Fireworks AI Streaming ASR v2 Setup

## Overview

The agent now uses **Fireworks AI Streaming ASR v2** as the primary Speech-to-Text (STT) engine, with **OpenAI GPT-4o mini** as a fallback. This provides:

- ⚡ **Lower latency**: Real-time streaming transcription
- 🎯 **Higher accuracy**: Better performance in noisy environments
- 💰 **Cost efficiency**: Fireworks offers competitive pricing
- 🔄 **Automatic fallback**: Seamlessly switches to OpenAI if Fireworks is unavailable

## Setup Instructions

### 1. Get Fireworks API Key

1. Go to [Fireworks AI](https://fireworks.ai/)
2. Sign up or log in to your account
3. Navigate to the API section
4. Generate a new API key

### 2. Update Environment Variables

Add your Fireworks API key to `.env.local`:

```env
# STT Configuration
# Primary: Fireworks AI Streaming ASR v2 (fast, accurate real-time transcription)
# Fallback: OpenAI GPT-4o mini (if Fireworks is unavailable or not configured)
FIREWORKS_API_KEY=your_actual_fireworks_api_key_here
```

### 3. Install Dependencies

```bash
uv pip install websocket-client
```

Or reinstall all dependencies:

```bash
uv sync
```

## How It Works

### STT Flow

1. **Primary**: Audio is sent to Fireworks AI Streaming ASR v2
   - WebSocket connection to `wss://audio-streaming-v2.api.fireworks.ai`
   - Streams PCM 16-bit audio chunks in **real-time with proper timing delays**
   - Uses 200ms chunks (Fireworks recommended)
   - Authorization via Bearer token in URL query params
   - Receives continuous transcription updates
   
2. **Fallback**: If Fireworks fails or is not configured
   - Automatically falls back to OpenAI GPT-4o mini
   - Uses OpenAI's Whisper-based transcription API
   - Ensures uninterrupted service

### Key Implementation Details

**Following Fireworks Best Practices:**
- ✅ **Chunk Duration**: 200ms chunks (3200 samples at 16kHz)
- ✅ **Real-time Streaming**: `time.sleep(chunk_duration)` between chunks
- ✅ **Bearer Token Auth**: Authorization passed as query param in URL
- ✅ **Segment-based State**: Dynamic array management as per Fireworks docs
- ✅ **Proper Audio Conversion**: NumPy array → PCM int16 bytes
- ✅ **Non-blocking**: Threaded WebSocket with event-based completion
   - Receives continuous transcription updates
2. **Fallback**: If Fireworks fails or is not configured
   - Automatically falls back to OpenAI GPT-4o mini
   - Uses OpenAI's Whisper-based transcription API
   - Ensures uninterrupted service

### Configuration Options

The STT system automatically determines which service to use:

- **If `FIREWORKS_API_KEY` is set**: Uses Fireworks as primary, OpenAI as fallback
- **If `FIREWORKS_API_KEY` is not set**: Uses OpenAI only

## Monitoring

The logs will show which STT service is being used:

```
🎤 STT (Fireworks): 'user transcription here'
```

Or if fallback is used:

```
Fireworks STT failed, falling back to OpenAI
🎤 STT (OpenAI fallback): 'user transcription here'
```

## Fireworks Models

Fireworks AI provides 4 voice models:

1. **whisper-v3**: Highest accuracy (pre-recorded)
2. **whisper-v3-turbo**: Faster processing (pre-recorded)
3. **Streaming ASR v1**: Real-time transcription (production-ready)
4. **Streaming ASR v2**: Next-gen real-time transcription (lower latency, higher accuracy) ✅ **WE USE THIS**

## Supported Languages

Fireworks Streaming ASR v2 supports 95+ languages including:

- English (en)
- Hindi (hi)
- Urdu (ur)
- Spanish (es)
- French (fr)
- German (de)
- Japanese (ja)
- Chinese (zh)
- Arabic (ar)
- And many more...

See [full language list](https://docs.fireworks.ai/api-reference/audio-transcriptions#supported-languages)

## Technical Implementation

### Audio Format Requirements
- **Sample Rate**: 16 kHz
- **Encoding**: PCM 16-bit little-endian  
- **Channels**: Mono
- **Chunk Size**: 200ms (3200 bytes = 1600 samples)
- **Streaming**: Real-time with timing delays between chunks

### Performance Improvements

**Original Issues (FIXED):**
- ❌ Sent all audio chunks at once (no streaming delay)
- ❌ Used HTTP headers for auth instead of query params
- ❌ Wrong auth format (plain key vs Bearer token)
- ❌ 100ms chunks instead of recommended 200ms
- ❌ Synchronous blocking approach

**Current Implementation:**
- ✅ Proper real-time streaming with `time.sleep(0.2)` between chunks
- ✅ Bearer token in URL query params (Fireworks standard)
- ✅ 200ms chunks as per Fireworks documentation
- ✅ Event-based WebSocket completion
- ✅ NumPy array handling for accurate audio conversion
- ✅ Matches Fireworks playground performance

### Transcription State Management

The Fireworks API uses a segment-based approach:

- Segments are identified by unique IDs
- Text is updated as transcription improves
- Client maintains dynamic array of segments
- Final transcription is assembled from all segments

## Troubleshooting

### Slow Performance / Bad Transcription

**If Fireworks feels slow:**
1. ✅ **FIXED**: Now using 200ms chunks with real-time delays
2. ✅ **FIXED**: Bearer token auth in URL (not headers)
3. ✅ **FIXED**: Proper streaming simulation
4. Should now match Fireworks playground speed

### Fireworks Connection Issues

If you see connection errors:

1. Check your API key is valid
2. Verify internet connectivity
3. Check Fireworks service status
4. The system will automatically fall back to OpenAI

### No Transcription Output

If neither service produces output:

1. Check audio input is working (VAD should detect speech)
2. Verify both API keys are set correctly
3. Check logs for specific error messages

### Latency Issues

If transcription feels slow:

1. Fireworks v2 should be faster than OpenAI
2. Check network connection
3. Verify chunk size settings are optimal

## Cost Considerations

- **Fireworks**: Check [Fireworks Pricing](https://fireworks.ai/pricing) for current rates
- **OpenAI**: Check [OpenAI Pricing](https://openai.com/pricing) for Whisper API rates

Using Fireworks as primary can be more cost-effective for high-volume usage.

## References

- [Fireworks Streaming ASR v2 Documentation](https://docs.fireworks.ai/api-reference/audio-streaming-transcriptions)
- [Fireworks ASR Models Guide](https://docs.fireworks.ai/guides/querying-asr-models)
- [Fireworks Python Cookbook](https://github.com/fw-ai/cookbook/tree/main/learn/audio/audio_streaming_speech_to_text)
