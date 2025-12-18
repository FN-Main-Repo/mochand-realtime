"""
LiveKit Pipeline for STT → LLM → TTS processing.
Optimized for smooth, natural-sounding audio output.
"""

import asyncio
import logging
import os
import io
import struct
from collections import deque
import re
import time
from typing import Optional, AsyncGenerator

import wave
from dotenv import load_dotenv
from openai import AsyncOpenAI
from google.cloud import texttospeech
from langdetect import detect, LangDetectException

from summarise_agent import summarize_text

load_dotenv(".env.local")

logger = logging.getLogger("livekit-pipeline")


def strip_markdown(text: str) -> str:
    """Remove all markdown formatting from text before speaking."""
    if not text:
        return text

    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


class AudioSmoother:
    """
    Handles audio crossfading and smoothing between chunks.
    Eliminates clicks/pops at chunk boundaries.
    """
    
    def __init__(self, sample_rate: int = 16000, crossfade_ms: int = 30):
        self.sample_rate = sample_rate
        self.crossfade_samples = int(sample_rate * crossfade_ms / 1000)
        self.previous_tail: Optional[bytes] = None
    
    def _bytes_to_samples(self, audio_bytes: bytes) -> list[int]:
        """Convert PCM bytes to list of 16-bit samples."""
        return list(struct.unpack(f'<{len(audio_bytes)//2}h', audio_bytes))
    
    def _samples_to_bytes(self, samples: list[int]) -> bytes:
        """Convert list of samples back to PCM bytes."""
        # Clamp to 16-bit range
        clamped = [max(-32768, min(32767, int(s))) for s in samples]
        return struct.pack(f'<{len(clamped)}h', *clamped)
    
    def process(self, audio_chunk: bytes) -> bytes:
        """
        Process audio chunk with crossfade from previous chunk.
        Call this sequentially for each chunk.
        """
        if len(audio_chunk) < self.crossfade_samples * 4:
            # Chunk too small for crossfade, return as-is
            return audio_chunk
        
        samples = self._bytes_to_samples(audio_chunk)
        
        # Apply fade-in to start
        for i in range(min(self.crossfade_samples, len(samples))):
            fade = i / self.crossfade_samples
            samples[i] = int(samples[i] * fade)
        
        # Crossfade with previous tail if available
        if self.previous_tail:
            tail_samples = self._bytes_to_samples(self.previous_tail)
            for i in range(min(len(tail_samples), len(samples), self.crossfade_samples)):
                # Blend previous tail (fading out) with current (fading in)
                fade_out = 1.0 - (i / self.crossfade_samples)
                fade_in = i / self.crossfade_samples
                samples[i] = int(tail_samples[i] * fade_out + samples[i] * fade_in)
        
        # Store tail for next chunk (with fade-out applied)
        if len(samples) > self.crossfade_samples:
            tail = samples[-self.crossfade_samples:]
            for i in range(len(tail)):
                fade = 1.0 - (i / len(tail))
                tail[i] = int(tail[i] * fade)
            self.previous_tail = self._samples_to_bytes(tail)
            
            # Return without the tail (will be crossfaded into next chunk)
            output = samples[:-self.crossfade_samples // 2]
        else:
            self.previous_tail = None
            output = samples
        
        return self._samples_to_bytes(output)
    
    def flush(self) -> Optional[bytes]:
        """Return any remaining audio (call after last chunk)."""
        if self.previous_tail:
            tail = self.previous_tail
            self.previous_tail = None
            return tail
        return None
    
    def reset(self):
        """Reset state for new utterance."""
        self.previous_tail = None


class LiveKitPipeline:
    """
    Standalone LiveKit-style pipeline for voice processing.
    Flow: Audio (PCM) → STT → LLM → TTS → Audio (PCM)
    """

    # Chunking configuration - LARGER chunks for smoother audio
    MIN_CHUNK_LENGTH = 100      # Increased from 50
    MAX_CHUNK_LENGTH = 300      # Increased from 180
    IDEAL_CHUNK_LENGTH = 200    # Target size
    MIN_AUDIO_SIZE = 1000

    SUPPORTED_LANGUAGES = {
        "en", "hi", "ur", "bn", "mr", "ta", "te", "gu", "kn", "ml", "pa",
        "es", "fr", "de", "pt", "it", "ja", "ko", "zh-cn", "zh-tw", "ar", "ru"
    }

    VOICE_MAP = {
        "en": ("en-IN-Standard-A", "en-IN"),
        "hi": ("hi-IN-Standard-A", "hi-IN"),
        "ur": ("ur-IN-Standard-A", "ur-IN"),
        "bn": ("bn-IN-Standard-A", "bn-IN"),
        "mr": ("mr-IN-Standard-A", "mr-IN"),
        "ta": ("ta-IN-Standard-A", "ta-IN"),
        "te": ("te-IN-Standard-A", "te-IN"),
        "gu": ("gu-IN-Standard-A", "gu-IN"),
        "kn": ("kn-IN-Standard-A", "kn-IN"),
        "ml": ("ml-IN-Standard-A", "ml-IN"),
        "pa": ("pa-IN-Standard-A", "pa-IN"),
        "es": ("es-ES-Standard-A", "es-ES"),
        "fr": ("fr-FR-Standard-A", "fr-FR"),
        "de": ("de-DE-Standard-A", "de-DE"),
        "pt": ("pt-BR-Standard-A", "pt-BR"),
        "it": ("it-IT-Standard-A", "it-IT"),
        "ja": ("ja-JP-Standard-B", "ja-JP"),
        "ko": ("ko-KR-Standard-A", "ko-KR"),
        "zh-cn": ("cmn-CN-Standard-A", "cmn-CN"),
        "zh-tw": ("cmn-TW-Standard-A", "cmn-TW"),
        "ar": ("ar-XA-Standard-A", "ar-XA"),
        "ru": ("ru-RU-Standard-A", "ru-RU"),
    }

    SYSTEM_PROMPT = """You are Jarvis, an intelligent AI assistant in a professional Google Meet call.

INSTRUCTIONS:
- Respond in the SAME language as the user speaks
- Be professional, helpful, and comprehensive
- For complex questions, provide thorough explanations
- For simple questions, keep it concise
- Maintain conversation context

NEVER include:
❌ URLs, links, web addresses, website names
❌ Markdown formatting like [text](url)
❌ Citations, references, or source attributions
❌ Phrases like "according to X" or "source: Y"

Provide natural, conversational responses in the user's language."""

    def __init__(self):
        """Initialize the pipeline with STT, LLM, and TTS clients."""
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.openrouter_client = AsyncOpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )

        self.tts_client = texttospeech.TextToSpeechClient.from_service_account_file(
            os.getenv("GOOGLE_CREDENTIALS_FILE")
        )

        self.tts_audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            # These settings help with smoother audio
            speaking_rate=1.0,
            pitch=0.0,
        )

        self.conversation_summary = ""
        self._assistant_turns = 0
        self._last_summarized_turn = 0
        self._summary_task: Optional[asyncio.Task] = None
        self._summary_lock = asyncio.Lock()

        self.messages: deque = deque(
            [{"role": "system", "content": self.SYSTEM_PROMPT}],
            maxlen=6
        )

        # Audio smoother for crossfading
        self.audio_smoother = AudioSmoother(sample_rate=16000, crossfade_ms=30)

        logger.info("LiveKit pipeline initialized")

    # ------------------------------------------------------------------
    # Text Cleaning
    # ------------------------------------------------------------------

    def _clean_for_tts(self, text: str) -> str:
        """Clean text for TTS synthesis."""
        clean = strip_markdown(text)
        clean = re.sub(r'https?://\S+', '', clean)
        clean = re.sub(r'\b[\w-]+\.(com|org|net|io|dev|ai)\b', '', clean)
        clean = re.sub(r'[*_~`#]', '', clean)
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()

    # ------------------------------------------------------------------
    # Chunking Logic - Optimized for natural speech
    # ------------------------------------------------------------------

    def _find_split_point(self, text: str) -> Optional[int]:
        """
        Find the best point to split text for TTS.
        Prioritizes complete sentences for natural flow.
        """
        length = len(text)
        
        if length < self.MIN_CHUNK_LENGTH:
            return None

        # Only split if we have enough text
        # Look for sentence endings first
        sentence_endings = '.?!।。؟'
        
        # Find ALL sentence endings in the valid range
        candidates = []
        for i, char in enumerate(text):
            if char in sentence_endings:
                # Prefer splits closer to IDEAL_CHUNK_LENGTH
                if self.MIN_CHUNK_LENGTH <= i + 1 <= self.MAX_CHUNK_LENGTH:
                    candidates.append(i + 1)
        
        if candidates:
            # Pick the one closest to ideal length
            ideal = self.IDEAL_CHUNK_LENGTH
            best = min(candidates, key=lambda x: abs(x - ideal))
            return best

        # If no sentence boundary found and we're at max, find any break
        if length >= self.MAX_CHUNK_LENGTH:
            # Try clause markers
            for marker in [',', ';', ':', '—', '–', ' - ']:
                idx = text.rfind(marker, self.MIN_CHUNK_LENGTH, self.MAX_CHUNK_LENGTH)
                if idx > 0:
                    return idx + 1
            
            # Fall back to word boundary
            idx = text.rfind(' ', self.MIN_CHUNK_LENGTH, self.MAX_CHUNK_LENGTH)
            if idx > 0:
                return idx + 1
            
            # Last resort: hard cut
            return self.MAX_CHUNK_LENGTH

        return None

    # ------------------------------------------------------------------
    # Voice Selection
    # ------------------------------------------------------------------

    def _select_voice_for_language(self, language_code: str) -> tuple[str, str]:
        """Resolve TTS voice for a given language."""
        if language_code in self.VOICE_MAP:
            return self.VOICE_MAP[language_code]

        base_lang = language_code.split("-")[0]
        if base_lang in self.VOICE_MAP:
            return self.VOICE_MAP[base_lang]

        logger.warning(f"⚠️ No voice for '{language_code}', using English")
        return ("en-IN-Standard-A", "en-IN")

    def _detect_language(self, text: str) -> str:
        """Detect language from text."""
        try:
            detected = detect(text)
            if detected == "zh":
                detected = "zh-cn"
            return detected if detected in self.SUPPORTED_LANGUAGES else "en"
        except LangDetectException:
            return "en"

    # ------------------------------------------------------------------
    # STT
    # ------------------------------------------------------------------

    async def transcribe_audio(self, audio_pcm: bytes, sample_rate: int = 16000) -> Optional[str]:
        """Transcribe audio using OpenAI Whisper."""
        try:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_pcm)

            wav_buffer.seek(0)
            wav_buffer.name = "audio.wav"

            transcript = await self.openai_client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=wav_buffer,
            )

            text = transcript.text.strip()
            if text:
                logger.info(f"🎤 STT: '{text}'")
                return text

            return None

        except Exception as e:
            logger.error(f"STT error: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    async def generate_response(self, user_text: str) -> str:
        """Generate LLM response (non-streaming)."""
        try:
            self.messages.append({"role": "user", "content": user_text})

            response = await self.openrouter_client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct",
                messages=list(self.messages),
                max_tokens=500,
                temperature=0.7,
            )

            assistant_text = response.choices[0].message.content.strip()
            clean_text = self._clean_for_tts(assistant_text)

            self.messages.append({"role": "assistant", "content": clean_text})

            logger.info(f"🤖 LLM: '{clean_text[:100]}...'")
            return clean_text

        except Exception as e:
            logger.error(f"LLM error: {e}", exc_info=True)
            return "I'm sorry, I encountered an error processing your request."

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    async def synthesize_speech(self, text: str, language: str, voice: str) -> bytes:
        """Convert text to speech using Google Cloud TTS."""
        try:
            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language,
                name=voice
            )

            synthesis_input = texttospeech.SynthesisInput(text=text)

            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=self.tts_audio_config
            )

            logger.debug(f"🔊 TTS: {len(response.audio_content)} bytes for {len(text)} chars")
            return response.audio_content

        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            return b'\x00' * 16000 * 2

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    async def _update_conversation_summary(self):
        """Update conversation summary using summarization agent."""
        async with self._summary_lock:
            try:
                full_conversation = "\n".join(
                    f"{msg['role'].capitalize()}: {msg['content']}"
                    for msg in self.messages
                    if msg['role'] in {'user', 'assistant'}
                )

                logger.info("📝 Updating conversation summary...")
                summary = await summarize_text(full_conversation)

                if summary:
                    self.conversation_summary += summary
                    logger.info(f"📝 Summary updated")

            except Exception as e:
                logger.error(f"📝 Summary error: {e}", exc_info=True)

    def _schedule_summary_update(self):
        """Schedule background summary update."""
        if self._summary_task and not self._summary_task.done():
            return
        self._summary_task = asyncio.create_task(self._update_conversation_summary())

    # ------------------------------------------------------------------
    # Full Pipeline (Non-streaming)
    # ------------------------------------------------------------------

    async def process_audio(self, audio_pcm: bytes, sample_rate: int = 16000) -> Optional[bytes]:
        """Full pipeline: Audio → STT → LLM → TTS → Audio"""
        text = await self.transcribe_audio(audio_pcm, sample_rate)
        if not text:
            return None

        response_text = await self.generate_response(text)
        language_code = self._detect_language(response_text)
        voice_name, lang_code = self._select_voice_for_language(language_code)

        return await self.synthesize_speech(response_text, lang_code, voice_name)

    # ------------------------------------------------------------------
    # Streaming Pipeline (from STT result)
    # ------------------------------------------------------------------

    async def process_audio_streaming(self, audio_pcm: bytes, sample_rate: int = 16000) -> AsyncGenerator[bytes, None]:
        """Streaming pipeline: Audio → STT → LLM streaming → TTS chunks"""
        text = await self.transcribe_audio(audio_pcm, sample_rate)

        if not text or len(text.strip()) < 2:
            logger.warning(f"⚠️ No valid transcription")
            return

        async for audio_chunk in self.process_audio_streaming_active(text):
            yield audio_chunk

    # ------------------------------------------------------------------
    # Active Streaming Pipeline (with smooth audio)
    # ------------------------------------------------------------------

    async def process_audio_streaming_active(self, user_text: str) -> AsyncGenerator[bytes, None]:
        """
        ACTIVE streaming pipeline with smooth audio.
        Uses larger chunks + crossfading for natural flow.
        """
        logger.info(f"🎙️ ACTIVE user: '{user_text[:60]}...'")

        # 1. Detect language and select voice ONCE
        language_code = self._detect_language(user_text)
        voice_name, language_code = self._select_voice_for_language(language_code)
        logger.info(f"🌍 Language: {language_code}, Voice: {voice_name}")

        # 2. Build message context
        self.messages.append({"role": "user", "content": user_text})

        messages = list(self.messages)
        messages.append({
            "role": "user",
            "content": f"Previous conversation: {self.conversation_summary}\n\nCurrent query: {user_text}"
        })

        # 3. Open LLM stream
        logger.info("🚀 Opening LLM stream")
        llm_start = time.time()

        stream = await self.openrouter_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            stream=True,
        )

        # 4. Reset audio smoother for new utterance
        self.audio_smoother.reset()

        # 5. Stream tokens → larger chunks → TTS → crossfade
        full_response = ""
        buffer = ""
        first_token = True
        chunks_synthesized = 0

        try:
            async for chunk in stream:
                if first_token:
                    logger.info(f"⏱️ First token: {time.time() - llm_start:.3f}s")
                    first_token = False

                delta = chunk.choices[0].delta.content
                if not delta:
                    continue

                full_response += delta
                buffer += delta

                clean_buffer = self._clean_for_tts(buffer)

                # Check for split point (using larger thresholds now)
                split_point = self._find_split_point(clean_buffer)

                if split_point:
                    chunk_text = clean_buffer[:split_point].strip()
                    remaining = clean_buffer[split_point:].strip()

                    if chunk_text and len(chunk_text) >= 10:
                        audio_chunk = await self.synthesize_speech(
                            text=chunk_text,
                            language=language_code,
                            voice=voice_name,
                        )

                        if audio_chunk and len(audio_chunk) > self.MIN_AUDIO_SIZE:
                            # Apply crossfade smoothing
                            smoothed = self.audio_smoother.process(audio_chunk)
                            chunks_synthesized += 1
                            logger.debug(f"🔊 Chunk {chunks_synthesized}: '{chunk_text[:50]}...'")
                            yield smoothed

                    buffer = remaining

            # 6. Flush remaining buffer
            if buffer.strip():
                clean_remainder = self._clean_for_tts(buffer)

                if clean_remainder and len(clean_remainder) >= 5:
                    audio_chunk = await self.synthesize_speech(
                        text=clean_remainder,
                        language=language_code,
                        voice=voice_name,
                    )

                    if audio_chunk and len(audio_chunk) > self.MIN_AUDIO_SIZE:
                        smoothed = self.audio_smoother.process(audio_chunk)
                        chunks_synthesized += 1
                        yield smoothed

            # Flush any remaining crossfade tail
            tail = self.audio_smoother.flush()
            if tail:
                yield tail

        except Exception:
            logger.error("❌ ACTIVE streaming failed", exc_info=True)
            raise

        # 7. Save assistant response
        clean_full = self._clean_for_tts(full_response)

        if clean_full:
            self.messages.append({"role": "assistant", "content": clean_full})
            self._assistant_turns += 1

            if self._assistant_turns - self._last_summarized_turn >= 6:
                self._last_summarized_turn = self._assistant_turns
                self._schedule_summary_update()

        logger.info(f"✅ Complete: {chunks_synthesized} chunks in {time.time() - llm_start:.2f}s")

    # ------------------------------------------------------------------
    # Batch Mode - Collect full response then synthesize (smoothest)
    # ------------------------------------------------------------------

    async def process_audio_batch(self, user_text: str) -> AsyncGenerator[bytes, None]:
        """
        BATCH mode: Collect full LLM response, then synthesize in optimal chunks.
        Higher latency but smoothest audio quality.
        """
        logger.info(f"🎙️ BATCH user: '{user_text[:60]}...'")

        language_code = self._detect_language(user_text)
        voice_name, language_code = self._select_voice_for_language(language_code)

        self.messages.append({"role": "user", "content": user_text})

        messages = list(self.messages)
        messages.append({
            "role": "user",
            "content": f"Previous conversation: {self.conversation_summary}\n\nCurrent query: {user_text}"
        })

        # Collect full response
        llm_start = time.time()
        
        stream = await self.openrouter_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            stream=True,
        )

        full_response = ""
        first_token = True
        
        async for chunk in stream:
            if first_token:
                logger.info(f"⏱️ First token: {time.time() - llm_start:.3f}s")
                first_token = False
                
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta

        logger.info(f"⏱️ Full response in: {time.time() - llm_start:.2f}s")

        # Clean and split into optimal chunks
        clean_response = self._clean_for_tts(full_response)
        
        if not clean_response:
            return

        # Split into sentence-based chunks
        chunks = self._split_into_sentences(clean_response)
        
        self.audio_smoother.reset()
        
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text) < 5:
                continue
                
            audio_chunk = await self.synthesize_speech(
                text=chunk_text,
                language=language_code,
                voice=voice_name,
            )

            if audio_chunk and len(audio_chunk) > self.MIN_AUDIO_SIZE:
                smoothed = self.audio_smoother.process(audio_chunk)
                logger.debug(f"🔊 Batch chunk {i+1}/{len(chunks)}")
                yield smoothed

        # Flush tail
        tail = self.audio_smoother.flush()
        if tail:
            yield tail

        # Save response
        self.messages.append({"role": "assistant", "content": clean_response})
        self._assistant_turns += 1

        logger.info(f"✅ BATCH complete: {len(chunks)} chunks")

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences, combining short ones."""
        # Split on sentence boundaries
        parts = re.split(r'(?<=[.?!।。])\s+', text)
        
        chunks = []
        current = ""
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            if not current:
                current = part
            elif len(current) + len(part) + 1 < self.IDEAL_CHUNK_LENGTH:
                # Combine short sentences
                current += " " + part
            else:
                if current:
                    chunks.append(current)
                current = part
        
        if current:
            chunks.append(current)
        
        return chunks

    # ------------------------------------------------------------------
    # Concurrent Streaming (lowest latency, good smoothness)
    # ------------------------------------------------------------------

    async def process_audio_streaming_concurrent(self, user_text: str) -> AsyncGenerator[bytes, None]:
        """
        Concurrent streaming with producer-consumer pattern.
        Good balance of latency and smoothness.
        """
        logger.info(f"🎙️ CONCURRENT user: '{user_text[:60]}...'")

        language_code = self._detect_language(user_text)
        voice_name, language_code = self._select_voice_for_language(language_code)

        self.messages.append({"role": "user", "content": user_text})

        messages = list(self.messages)
        messages.append({
            "role": "user",
            "content": f"Previous conversation: {self.conversation_summary}\n\nCurrent query: {user_text}"
        })

        text_queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=5)
        audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=3)

        full_response = ""
        llm_start = time.time()

        async def llm_producer():
            nonlocal full_response
            buffer = ""

            try:
                stream = await self.openrouter_client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=messages,
                    stream=True,
                )

                first_token = True
                async for chunk in stream:
                    if first_token:
                        logger.info(f"⏱️ First token: {time.time() - llm_start:.3f}s")
                        first_token = False

                    delta = chunk.choices[0].delta.content
                    if not delta:
                        continue

                    full_response += delta
                    buffer += delta
                    clean = self._clean_for_tts(buffer)

                    split_point = self._find_split_point(clean)
                    if split_point:
                        await text_queue.put(clean[:split_point].strip())
                        buffer = clean[split_point:]

                if buffer.strip():
                    clean = self._clean_for_tts(buffer)
                    if clean:
                        await text_queue.put(clean)

            except Exception as e:
                logger.error(f"❌ LLM producer error: {e}")
            finally:
                await text_queue.put(None)

        async def tts_consumer():
            smoother = AudioSmoother(sample_rate=16000, crossfade_ms=30)
            
            try:
                while True:
                    text = await text_queue.get()
                    if text is None:
                        break

                    if len(text) < 5:
                        continue

                    try:
                        audio = await self.synthesize_speech(
                            text=text,
                            language=language_code,
                            voice=voice_name,
                        )

                        if audio and len(audio) > self.MIN_AUDIO_SIZE:
                            smoothed = smoother.process(audio)
                            await audio_queue.put(smoothed)

                    except Exception as e:
                        logger.warning(f"⚠️ TTS error: {e}")

                # Flush smoother
                tail = smoother.flush()
                if tail:
                    await audio_queue.put(tail)

            except Exception as e:
                logger.error(f"❌ TTS consumer error: {e}")
            finally:
                await audio_queue.put(None)

        producer_task = asyncio.create_task(llm_producer())
        consumer_task = asyncio.create_task(tts_consumer())

        try:
            while True:
                audio = await audio_queue.get()
                if audio is None:
                    break
                yield audio

        finally:
            await producer_task
            await consumer_task

        clean_full = self._clean_for_tts(full_response)
        if clean_full:
            self.messages.append({"role": "assistant", "content": clean_full})
            self._assistant_turns += 1

            if self._assistant_turns - self._last_summarized_turn >= 6:
                self._last_summarized_turn = self._assistant_turns
                self._schedule_summary_update()

        logger.info(f"✅ CONCURRENT complete: {time.time() - llm_start:.2f}s")