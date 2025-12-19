"""
Universal LLM client supporting multiple providers
Simplified version to avoid import issues
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def remove_double_quotes(text: str) -> str:
    """Utility to remove double quotes from text"""
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    return text

class LLMClient:
    """Universal async LLM client with multi-provider support"""
    
    def __init__(self, config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        await self.start_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_session()
    
    async def start_session(self):
        """Start HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=getattr(self.config, 'timeout', 30))
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def generate(self, messages: List[Dict[str, str]], 
                      temperature: float,                       # ✅ REQUIRED parameter
                      system_prompt: Optional[str] = None,
                      max_tokens: Optional[int] = None,
                      thinking: Optional[bool]=False) -> str:
        """Generate response using configured LLM"""
        
        logger.info(f"🤖 API call: {self.config.provider}/{self.config.model}")
        
        if not self.session:
            await self.start_session()
        
        temp = temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        try:
            if self.config.provider == 'anthropic':
                return await self._anthropic_request(messages, temp, tokens)
            elif self.config.provider == 'deepseek':
                return await self._deepseek_request(messages, temp, tokens)
            elif self.config.provider in ['openai', 'openrouter', 'groq']:
                return remove_double_quotes(await self._openai_compatible_request(messages, temp, tokens, thinking))
            elif self.config.provider == 'sarvam':
                return remove_double_quotes(await self._sarvam_compatible_request(messages, temp, tokens))
            else:
                raise Exception(f"Unsupported provider: {self.config.provider}")
        except Exception as e:
            logger.error(f"❌ 🤖 Generation failed: {type(e).__name__}: {str(e)}")
            raise Exception(f"Generation failed: {e}")
        
    
    async def _deepseek_request(self, messages: List[Dict[str, str]], 
                           temperature: float, max_tokens: int) -> str:
        """Handle Deepseek API requests"""
        
        
        # API configuration
        api_url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        # Request payload
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # Make async request
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"]
            
        
        
    
    async def _anthropic_request(self, messages: List[Dict[str, str]], 
                               temperature: float, max_tokens: int) -> str:
        """Handle Anthropic API requests"""
        
        system_content = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)
        
        headers = {
            "x-api-key": self.config.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages
        }
        
        if system_content:
            payload["system"] = system_content
        
        async with self.session.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        ) as response:
            
            logger.info(f"🤖 Anthropic response status: {response.status}")
            
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"❌ 🤖 Anthropic API error: {response.status}: {error_text}")
                raise Exception(f"API error {response.status}: {error_text}")
            
            result = await response.json()
            return result["content"][0]["text"]
    
    async def _sarvam_compatible_request(self, messages: List[Dict[str, str]], 
                                       temperature: float, max_tokens: int) -> str:
        """Handle Sarvam-compatible API requests"""
        
        url  = "https://api.sarvam.ai/v1/chat/completions"
        headers = {
            "api-subscription-key": self.config.api_key, 
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                
                logger.info(f"🤖 Sarvam response status: {response.status}")
                
                response_text = await response.text()
                
                if response.status != 200:
                    logger.error(f"❌ 🤖 Sarvam API error: {response.status}: {response_text}")
                    raise Exception(f"API error {response.status}: {response_text}")
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]
    
    async def _openai_compatible_request(self, messages: List[Dict[str, str]], 
                                       temperature: float, max_tokens: int, thinking:bool) -> str:
        """Handle OpenAI-compatible API requests"""
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        if self.config.provider == 'openrouter':
            headers["HTTP-Referer"] = "https://github.com/brain-heart-research"
            headers["X-Title"] = "Brain-Heart Research System"
        
        if thinking:
            logger.info(f"🧠 Thinking mode enabled for {self.config.provider} model {self.config.model}")
            payload = {
                "model": self.config.model,
                "messages": messages,
                "provider": {
                'sort': 'throughput'
                },
                "temperature": temperature,
                "max_tokens": max_tokens,
                "reasoning": {
                    "max_tokens": 2000
                }
            }
        else:
            payload = {
                "model": self.config.model,
                "messages": messages,
                "provider": {
                'sort': 'throughput'
                },
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        
        if hasattr(self.config, 'base_url') and self.config.base_url:
            url = f"{self.config.base_url}/chat/completions"
        else:
            url = "https://api.openai.com/v1/chat/completions"
        
        async with self.session.post(url, headers=headers, json=payload) as response:
            
            logger.info(f"🤖 {self.config.provider} response status: {response.status}")
            
            response_text = await response.text()
            
            if response.status != 200:
                logger.error(f"❌ 🤖 {self.config.provider} API error: {response.status}: {response_text}")
                raise Exception(f"API error {response.status}: {response_text}")
            
            # Safe JSON parsing with better error handling
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"❌ 🤖 JSON parse error: {response_text[:100]}...")
                raise Exception(f"Invalid JSON from {self.config.provider}: '{response_text[:200]}...' Error: {e}")
            
            # Extract content - handle thinking models that use 'reasoning' field
            choice = result["choices"][0]
            message = choice["message"]
            content = message.get("content", "")
            
            # Thinking models (like Qwen) put reasoning in separate field
            if (content is None or content == "") and "reasoning" in message:
                reasoning = message["reasoning"]
                logger.info(f"🧠 Thinking model detected - using 'reasoning' field")
                
                # Show the FULL reasoning/thinking process
                print(f"\n{'='*80}")
                print(f"💭 FULL THINKING PROCESS (RAW):")
                print(f"{'='*80}")
                print(reasoning)
                print(f"{'='*80}\n")
                
                content = reasoning
            elif "reasoning" in message and message.get("content"):
                # Both fields exist - show reasoning separately
                reasoning = message["reasoning"]
                logger.info(f"🧠 Thinking model with both fields")
                
                print(f"\n{'='*80}")
                print(f"💭 FULL THINKING PROCESS (RAW):")
                print(f"{'='*80}")
                print(reasoning)
                print(f"{'='*80}\n")
            
            # Check if we hit token limit
            if choice.get("finish_reason") == "length":
                logger.warning(f"⚠️ Response truncated due to token limit!")
                logger.warning(f"   Current max_tokens: {max_tokens}")
                logger.warning(f"   Consider increasing max_tokens in config")
            
            if not content:
                logger.error(f"❌ No content found. Message keys: {message.keys()}")
                raise Exception(f"Empty content in response")
            
            return content
    
    def get_model_info(self) -> Dict[str, str]:
        """Get model information"""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "max_tokens": str(self.config.max_tokens)
        }
