import httpx
import asyncio
from os import getenv
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_URL = "https://api.groq.com/openai/v1/chat/completions"

async def summarize_text(text, model="meta-llama/llama-3.3-70b-instruct"):
    """
    Summarizes the given text using OpenRouter's GPT model.

    Args:
        text (str): The text to summarize.
        model (str): The model to use for summarization. Default is "gpt-3.5-turbo".

    Returns:
        str: The summarized text.
    """
    headers = {
        "Authorization": f"Bearer {getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Summarize the following text:\n{text}"}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            summary = data['choices'][0]['message']['content']
            return summary.strip()
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None


async def main():
    input_text = """
    Artificial intelligence (AI) is a branch of computer science that aims to create machines 
    that can perform tasks that would typically require human intelligence. These tasks include 
    learning, reasoning, problem-solving, perception, and language understanding. AI is being 
    applied in various fields such as healthcare, finance, education, and transportation.
    """
    summary = await summarize_text(input_text)
    print("Summary:")
    print(summary)

if __name__ == "__main__":
    asyncio.run(main())