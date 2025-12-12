"""
Audio format conversion utilities for Google Meet integration.
Handles conversion between different audio formats and sample rates.
"""

import base64
import logging
import struct
import numpy as np
from typing import Optional

logger = logging.getLogger("audio-transport")


class AudioTransport:
    """Handles audio format conversions for Attendee.dev and LiveKit integration."""

    @staticmethod
    def base64_to_pcm(base64_data: str) -> bytes:
        """
        Convert base64-encoded audio data to raw PCM bytes.
        
        Args:
            base64_data: Base64-encoded audio string
            
        Returns:
            Raw PCM audio bytes
        """
        return base64.b64decode(base64_data)

    @staticmethod
    def pcm_to_base64(pcm_data: bytes) -> str:
        """
        Convert raw PCM bytes to base64-encoded string.
        
        Args:
            pcm_data: Raw PCM audio bytes
            
        Returns:
            Base64-encoded audio string
        """
        return base64.b64encode(pcm_data).decode('utf-8')

    @staticmethod
    def resample_audio(
        audio_data: bytes,
        src_sample_rate: int,
        dst_sample_rate: int,
        num_channels: int = 1
    ) -> bytes:
        """
        Resample audio from one sample rate to another.
        
        Args:
            audio_data: Raw PCM audio bytes (16-bit)
            src_sample_rate: Source sample rate in Hz
            dst_sample_rate: Destination sample rate in Hz
            num_channels: Number of audio channels (1 for mono, 2 for stereo)
            
        Returns:
            Resampled PCM audio bytes
        """
        if src_sample_rate == dst_sample_rate:
            return audio_data

        # Convert bytes to numpy array of int16
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate resampling ratio
        ratio = dst_sample_rate / src_sample_rate
        
        # Calculate new length
        new_length = int(len(audio_array) * ratio)
        
        # Resample using numpy interpolation
        resampled = np.interp(
            np.linspace(0, len(audio_array) - 1, new_length),
            np.arange(len(audio_array)),
            audio_array
        ).astype(np.int16)
        
        return resampled.tobytes()

    @staticmethod
    def convert_float32_to_int16(float32_data: bytes) -> bytes:
        """
        Convert 32-bit float PCM to 16-bit int PCM.
        
        Args:
            float32_data: Raw PCM audio bytes (32-bit float)
            
        Returns:
            16-bit int PCM audio bytes
        """
        # Convert bytes to numpy array of float32
        float_array = np.frombuffer(float32_data, dtype=np.float32)
        
        # Clip values to [-1.0, 1.0] and convert to int16
        int16_array = (np.clip(float_array, -1.0, 1.0) * 32767).astype(np.int16)
        
        return int16_array.tobytes()

    @staticmethod
    def convert_int16_to_float32(int16_data: bytes) -> bytes:
        """
        Convert 16-bit int PCM to 32-bit float PCM.
        
        Args:
            int16_data: Raw PCM audio bytes (16-bit int)
            
        Returns:
            32-bit float PCM audio bytes
        """
        # Convert bytes to numpy array of int16
        int16_array = np.frombuffer(int16_data, dtype=np.int16)
        
        # Convert to float32 in range [-1.0, 1.0]
        float32_array = int16_array.astype(np.float32) / 32767.0
        
        return float32_array.tobytes()

    @staticmethod
    def chunk_audio(
        audio_data: bytes,
        chunk_duration_ms: int,
        sample_rate: int,
        bytes_per_sample: int = 2
    ) -> list[bytes]:
        """
        Split audio data into chunks of specified duration.
        
        Args:
            audio_data: Raw PCM audio bytes
            chunk_duration_ms: Duration of each chunk in milliseconds
            sample_rate: Sample rate in Hz
            bytes_per_sample: Bytes per sample (2 for 16-bit, 4 for 32-bit)
            
        Returns:
            List of audio chunks
        """
        # Calculate chunk size in bytes
        samples_per_chunk = int((chunk_duration_ms / 1000.0) * sample_rate)
        bytes_per_chunk = samples_per_chunk * bytes_per_sample
        
        chunks = []
        for i in range(0, len(audio_data), bytes_per_chunk):
            chunk = audio_data[i:i + bytes_per_chunk]
            if len(chunk) > 0:
                chunks.append(chunk)
        
        return chunks

    @staticmethod
    def create_attendee_audio_message(
        pcm_data: bytes,
        sample_rate: int = 16000
    ) -> dict:
        """
        Create an Attendee.dev-compatible audio output message.
        
        Args:
            pcm_data: Raw PCM audio bytes (16-bit, mono)
            sample_rate: Sample rate in Hz (8000, 16000, or 24000)
            
        Returns:
            Dictionary in Attendee.dev message format
        """
        base64_chunk = AudioTransport.pcm_to_base64(pcm_data)
        
        return {
            "trigger": "realtime_audio.bot_output",
            "data": {
                "chunk": base64_chunk,
                "sample_rate": sample_rate
            }
        }

    @staticmethod
    def parse_attendee_audio_message(message: dict) -> Optional[tuple[bytes, int, int]]:
        """
        Parse an Attendee.dev audio input message.
        
        Args:
            message: Attendee.dev message dictionary
            
        Returns:
            Tuple of (pcm_data, sample_rate, timestamp_ms) or None if invalid
        """
        try:
            if message.get("trigger") != "realtime_audio.mixed":
                logger.warning(f"Unknown message trigger: {message.get('trigger')}")
                return None
            
            data = message.get("data", {})
            base64_chunk = data.get("chunk")
            sample_rate = data.get("sample_rate", 16000)
            timestamp_ms = data.get("timestamp_ms", 0)
            
            if not base64_chunk:
                logger.warning("No chunk data in message")
                return None
            
            pcm_data = AudioTransport.base64_to_pcm(base64_chunk)
            
            return (pcm_data, sample_rate, timestamp_ms)
        except Exception as e:
            logger.error(f"Error parsing Attendee audio message: {e}")
            return None

    @staticmethod
    def get_audio_duration_ms(
        audio_data: bytes,
        sample_rate: int,
        bytes_per_sample: int = 2
    ) -> float:
        """
        Calculate the duration of audio data in milliseconds.
        
        Args:
            audio_data: Raw PCM audio bytes
            sample_rate: Sample rate in Hz
            bytes_per_sample: Bytes per sample (2 for 16-bit)
            
        Returns:
            Duration in milliseconds
        """
        num_samples = len(audio_data) // bytes_per_sample
        duration_seconds = num_samples / sample_rate
        return duration_seconds * 1000.0
