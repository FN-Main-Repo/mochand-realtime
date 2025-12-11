import aiohttp
import json
import logging
import asyncio
from typing import Any, Literal
from livekit.agents import llm, utils
from livekit.agents.llm import ChatContext, ChatChunk, ChatMessage
from livekit.agents.inference.llm import LLMStream as _LLMStream
from livekit.agents.types import APIConnectOptions

# ChatRole is a Literal type, not an enum
ChatRole = Literal["system", "user", "assistant", "tool"]

logger = logging.getLogger("fastapi-llm")
logger.setLevel(logging.DEBUG)


def extract_text_from_content(content: Any) -> str:
    """Extract text from message content (handles list or string)"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Content is a list of text/image/audio items
        # Extract only text content
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif hasattr(item, 'text'):
                text_parts.append(item.text)
        return ' '.join(text_parts)
    else:
        return str(content)


class AgentLLM(llm.LLM):
    def __init__(
        self,
        api_url: str,
        userid: str = "voice_agent_user",
        mode: str = "brief",
        source: str = "voice",
        timeout: int = 30,
    ):
        super().__init__()
        self.api_url = api_url
        self.userid = userid
        self.mode = mode
        self.source = source
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        logger.info(f"Initialized FastAPILangChainLLM with URL: {api_url}")
    
    @property
    def model(self) -> str:
        """Return the model identifier for this LLM"""
        return "custom-fastapi-llm"
    
    @property
    def provider(self) -> str:
        """Return the provider identifier for this LLM"""
        return "custom-fastapi"
    
    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: Any = None,
        conn_options: APIConnectOptions = APIConnectOptions(),
        fnc_ctx: Any = None,
        temperature: float | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: str | None = None,
        extra_kwargs: Any = None,
    ) -> "LLMStream":
        """
        Return an LLM stream that LiveKit can use
        """
        logger.info("Creating LLMStream for chat request")
        return LLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            fnc_ctx=fnc_ctx,
            temperature=temperature,
            extra_kwargs=extra_kwargs,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: AgentLLM,
        *,
        chat_ctx: ChatContext,
        tools: Any = None,
        conn_options: APIConnectOptions,
        fnc_ctx: Any = None,
        temperature: float | None = None,
        extra_kwargs: Any = None,
    ):
        """
        Initialize the LLM stream following LiveKit's exact pattern
        """
        logger.info("LLMStream.__init__ called")
        
        # Store custom attributes BEFORE calling super().__init__()
        self._llm_instance = llm
        self._temperature = temperature
        self._fnc_ctx = fnc_ctx
        self._extra_kwargs = extra_kwargs
        
        # Call parent __init__ with required parameters
        super().__init__(
            llm=llm,
            chat_ctx=chat_ctx,
            tools=tools if tools is not None else [],
            conn_options=conn_options,
        )
        
        logger.info("LLMStream initialization complete")
        
    async def _run(self) -> None:
        logger.info("=== _run() method started ===")

        request_id = f"req_{id(self)}_{asyncio.current_task().get_name() if asyncio.current_task() else 'unknown'}"

        try:
            chat_history = []
            user_query = ""

            items = self._chat_ctx.items
            logger.info(f"Processing {len(items)} items from chat context")

            messages = [item for item in items if hasattr(item, "role")]

            for i, msg in enumerate(messages):
                content_text = extract_text_from_content(msg.content)
                logger.debug(
                    f"Message {i}: role={msg.role}, content={content_text[:50] if content_text else 'empty'}..."
                )

                if i == len(messages) - 1 and msg.role == "user":
                    user_query = content_text
                    logger.info(f"User query: {user_query}")
                else:
                    if msg.role == "user":
                        chat_history.append({"role": "user", "content": content_text})
                    elif msg.role == "assistant":
                        chat_history.append({"role": "assistant", "content": content_text})

            payload = {
                "userid": self._llm_instance.userid,
                "chat_history": chat_history,
                "user_query": user_query,
                "mode": "brief",
                "source": "whatsapp",
            }

            if self._temperature is not None:
                payload["temperature"] = self._temperature

            if self._extra_kwargs:
                payload.update(self._extra_kwargs)

            logger.info(
                f"Prepared payload: userid={payload['userid']}, mode={payload['mode']}, query={user_query[:50] if user_query else 'empty'}"
            )
            logger.info(f"Making STREAM POST request to: {self._llm_instance.api_url}")

            async with aiohttp.ClientSession(timeout=self._llm_instance.timeout) as session:
                logger.info("HTTP session created, sending streaming request...")
                async with session.post(
                    self._llm_instance.api_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "accept": "application/json",
                    },
                ) as response:
                    logger.info(f"Got streaming response with status: {response.status}")
                    response.raise_for_status()

                    async for chunk_bytes in response.content:
                        if not chunk_bytes:
                            continue

                        text = chunk_bytes.decode("utf-8")
                        if not text.strip():
                            continue

                        chunk = ChatChunk(
                            id=request_id,
                            delta=llm.ChoiceDelta(
                                role="assistant",
                                content=text,
                            ),
                        )
                        self._event_ch.send_nowait(chunk)
                        logger.debug(f"Sent streamed chunk: {text!r}")

        except aiohttp.ClientError as e:
            logger.error(f"HTTP Error calling FastAPI streaming agent: {e}", exc_info=True)
            chunk = ChatChunk(
                id=request_id,
                delta=llm.ChoiceDelta(
                    role="assistant",
                    content="I'm sorry, I'm having trouble connecting right now.",
                ),
            )
            self._event_ch.send_nowait(chunk)
        except Exception as e:
            logger.error(f"Unexpected error in _run(): {e}", exc_info=True)
            chunk = ChatChunk(
                id=request_id,
                delta=llm.ChoiceDelta(
                    role="assistant",
                    content="I apologize, but something went wrong.",
                ),
            )
            self._event_ch.send_nowait(chunk)
