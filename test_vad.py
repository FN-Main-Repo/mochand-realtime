import asyncio
from livekit.plugins import silero

async def test():
    vad = silero.VAD.load()
    stream = vad.stream()
    print('VADStream methods:')
    methods = [m for m in dir(stream) if not m.startswith('_')]
    for m in methods:
        print(f"  - {m}")
    
    # Check if it's an async iterator
    print(f"\nIs async iterator: {hasattr(stream, '__aiter__')}")
    print(f"Has push_frame: {hasattr(stream, 'push_frame')}")
    
    await stream.aclose()

asyncio.run(test())
