"""
LiveKit Pipeline for STT → LLM → TTS processing.
Standalone pipeline that can be used without a LiveKit room.
"""

import asyncio
import logging
import os
import io
import numpy as np
from typing import Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI
from google.cloud import texttospeech
import wave
from langdetect import detect, LangDetectException

load_dotenv(".env.local")

logger = logging.getLogger("livekit-pipeline")


class LiveKitPipeline:
    """
    Standalone LiveKit-style pipeline for voice processing.
    
    Flow: Audio (PCM) → STT → LLM → TTS → Audio (PCM)
    """
    
    def __init__(self):
        """Initialize the pipeline with STT, LLM, and TTS clients."""
        # OpenAI for STT and LLM
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # OpenRouter for Gemini LLM
        self.openrouter_client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Google Cloud TTS
        self.tts_client = texttospeech.TextToSpeechClient.from_service_account_file(
            os.getenv("GOOGLE_CREDENTIALS_FILE")
        )
        
        # Conversation history for context
        self.conversation_history = []
        
        self.tts_audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000
        )
        
        # Conversation history
        # Conversation history
        self.messages = [
            {
                "role": "system",
                "content": """You are Jarvis, an intelligent AI assistant in a professional Google Meet call.
                
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
            }
        ]
        
        logger.info("LiveKit pipeline initialized")
    
    async def transcribe_audio(self, audio_pcm: bytes, sample_rate: int = 16000) -> Optional[str]:
        """
        Transcribe audio using OpenAI Whisper.
        
        Args:
            audio_pcm: Raw PCM audio bytes (16-bit, mono)
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed text or None if no speech detected
        """
        try:
            # Convert PCM to WAV format for Whisper API
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_pcm)
            
            wav_buffer.seek(0)
            wav_buffer.name = "audio.wav"
            
            # Call Whisper API
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
    
    async def generate_response(self, user_text: str) -> str:
        """
        Generate LLM response using Gemini via OpenRouter.
        
        Args:
            user_text: User's transcribed text
            
        Returns:
            LLM response text
        """
        try:
            # Add user message to history
            self.messages.append({
                "role": "user",
                "content": user_text
            })
            
            # Call Gemini via OpenRouter
            response = await self.openrouter_client.chat.completions.create(
                model="google/gemini-2.0-flash-001:online",
                messages=self.messages,
                max_tokens=500,
                temperature=0.7
            )
            
            assistant_text = response.choices[0].message.content.strip()
            
            # CRITICAL: Strip markdown links and URLs before speaking
            # Remove markdown links: [text](url) -> text
            import re
            clean_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', assistant_text)
            # Remove any remaining URLs
            clean_text = re.sub(r'https?://\S+', '', clean_text)
            # Remove website mentions like "accuweather.com"
            clean_text = re.sub(r'\b[a-zA-Z0-9-]+\.(com|org|net|io|dev|ai)\b', '', clean_text)
            # Clean up extra spaces
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            # Add CLEANED response to history (so future context is clean)
            self.messages.append({
                "role": "assistant",
                "content": clean_text
            })
            
            # Keep conversation history manageable (last 6 messages for faster processing)
            if len(self.messages) > 7:  # system + 6 messages
                self.messages = [self.messages[0]] + self.messages[-6:]
            
            logger.info(f"🤖 LLM (raw): '{assistant_text}'")
            if clean_text != assistant_text:
                logger.info(f"🤖 LLM (cleaned): '{clean_text}'")
            
            return clean_text
            
        except Exception as e:
            logger.error(f"LLM error: {e}", exc_info=True)
            return "I'm sorry, I encountered an error processing your request."
        
    async def generate_response_streaming(self, user_text: str):
        """Stream LLM response and yield chunks for TTS with language detection."""
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_text})
        
        # Build messages: system prompt + conversation history
        messages = self.messages.copy()  # Start with system prompt
        messages.extend(self.conversation_history)  # Add all conversation
        
        # Trim old messages if too long (keep system + last 6 messages)
        if len(messages) > 7:
            messages = [messages[0]] + messages[-6:]
        
        # Enable streaming
        stream = await self.openrouter_client.chat.completions.create(
            model="google/gemini-2.0-flash-001:online",
            messages=messages,
            max_tokens=16000,
            temperature=0.7,
            stream=True
        )
        
        full_response = ""
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
        
        # Detect language using langdetect
        from langdetect import detect, LangDetectException
        
        try:
            detected_lang = detect(full_response)
            
            # langdetect sometimes returns 'zh' instead of 'zh-cn' or 'zh-tw'
            # Default Chinese to simplified (zh-cn)
            if detected_lang == 'zh':
                detected_lang = 'zh-cn'
            
            # Check if detected language is supported in VOICE_MAP
            # (defined in synthesize_speech method)
            supported_languages = {
                "en", "hi", "ur", "bn", "mr", "ta", "te", "gu", "kn", "ml", "pa",
                "es", "fr", "de", "pt", "it", "ja", "ko", "zh-cn", "zh-tw", "ar", "ru"
            }
            
            if detected_lang in supported_languages:
                language_code = detected_lang
            else:
                # Unsupported language, default to English
                logger.info(f"🔍 Detected unsupported language '{detected_lang}', using English")
                language_code = "en"
            
            logger.info(f"🔍 Detected language: {language_code}")
        except LangDetectException:
            logger.warning("Could not detect language, defaulting to English")
            language_code = "en"
        
        # Save to conversation history
        self.conversation_history.append({"role": "assistant", "content": full_response})
        
        # Yield response text with language code
        yield {"text": full_response, "language": language_code}
    
    async def synthesize_speech(self, text: str, language: str = "en") -> bytes:
        """
        Convert text to speech using Google Cloud TTS with specified language.
        
        Args:
            text: Text to synthesize
            language: ISO 639-1 language code (from LLM detection)
            
        Returns:
            PCM audio bytes (16-bit, mono, 16kHz)
        """
        try:
            # Map language codes to Google Cloud TTS language codes and voices
            VOICE_MAP = {
                "en": ("en-IN-Standard-C", "en-IN"),
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
            
            voice_name, language_code = VOICE_MAP.get(language, ("en-IN-Standard-C", "en-IN"))
            
            # Use specified voice
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )
            
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Call TTS API
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=self.tts_audio_config
            )
            
            logger.info(f"🔊 TTS: Generated {len(response.audio_content)} bytes (lang: {language})")
            
            return response.audio_content
            
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            # Return silence on error
            return b'\x00' * 16000 * 2  # 1 second of silence
    
    async def process_audio(self, audio_pcm: bytes, sample_rate: int = 16000) -> Optional[bytes]:
        """
        Full pipeline: Audio → STT → LLM → TTS → Audio
        
        Args:
            audio_pcm: Input audio PCM bytes
            sample_rate: Audio sample rate
            
        Returns:
            Response audio PCM bytes or None if no speech detected
        """
        # Step 1: STT
        text = await self.transcribe_audio(audio_pcm, sample_rate)
        
        if not text:
            return None
        
        # Step 2: LLM
        response_text = await self.generate_response(text)
        
        # Step 3: TTS
        response_audio = await self.synthesize_speech(response_text)
        
        return response_audio
    
    async def process_audio_streaming(self, audio_pcm: bytes, sample_rate: int = 16000):
        """
        STREAMING pipeline: Audio → STT → LLM (streaming) → TTS (per chunk) → Audio chunks
        
        Args:
            audio_pcm: Input audio PCM bytes
            sample_rate: Audio sample rate
            
        Yields:
            Audio PCM chunks as they're generated
        """
        # Step 1: STT (still blocking, but fast with Whisper)
        text = await self.transcribe_audio(audio_pcm, sample_rate)
        
        if not text:
            return
        
        # Step 2 & 3: Stream LLM → TTS
        # Get response with language detection from LLM
        async for response_data in self.generate_response_streaming(text):
            response_text = response_data["text"]
            language_code = response_data["language"]
            
            # Clean the text before TTS
            import re
            clean_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', response_text)
            clean_text = re.sub(r'https?://\S+', '', clean_text)
            clean_text = re.sub(r'\b[a-zA-Z0-9-]+\.(com|org|net|io|dev|ai)\b', '', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if clean_text:
                logger.info(f"🔊 TTS chunk: {len(clean_text)} chars for '{clean_text[:100]}'")
                # Convert to speech with detected language
                audio_chunk = await self.synthesize_speech(clean_text, language_code)
                yield audio_chunk
