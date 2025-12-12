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
        
        # Voice configuration
        self.tts_voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        self.tts_audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000
        )
        
        # Conversation history
        self.messages = [
            {
                "role": "system",
                "content": """You are Jarvis, a helpful AI voice assistant in a Google Meet call. 
                You provide concise, natural responses suitable for voice conversation.
                Keep responses brief (1-2 sentences maximum).
                Be friendly, professional, and helpful.
                
                CRITICAL RULES:
                - NEVER include URLs, links, web addresses, or website names
                - NEVER use markdown formatting like [text](url) 
                - NEVER mention source websites like accuweather.com, cambridge.org, etc.
                - NEVER add citations or references
                - Only provide the information itself, not where it came from
                - Speak naturally as if in a conversation, not reading from sources"""
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
                max_tokens=100,  # Shorter responses = faster
                temperature=0.7,  # Slightly more focused responses
                extra_headers={
                    "HTTP-Referer": "https://mochand.com",
                    "X-Title": "Google Meet Voice Agent"
                }
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
    
    async def synthesize_speech(self, text: str) -> bytes:
        """
        Convert text to speech using Google Cloud TTS.
        
        Args:
            text: Text to synthesize
            
        Returns:
            PCM audio bytes (16-bit, mono, 16kHz)
        """
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Call TTS API (synchronous, but fast enough)
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=self.tts_voice,
                audio_config=self.tts_audio_config
            )
            
            # response.audio_content is already PCM 16-bit mono at 16kHz
            logger.info(f"🔊 TTS: Generated {len(response.audio_content)} bytes")
            
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
