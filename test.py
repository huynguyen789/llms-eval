import asyncio
from openai import AsyncOpenAI
import os

api = os.environ.get("DEEPSEEK_API_KEY")
if not api:
    print("fail")

async_client = AsyncOpenAI(api_key=api, base_url="https://api.deepseek.com")

async def get_completion(prompt: str):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = await async_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content, response.choices[0].message.reasoning_content
    except Exception as e:
        print(f"Error: {e}")
        return None, None

async def main():
    # Example single request
    content, reasoning = await get_completion("Jessica has 2 brothers and a sister. How many sisters each of her brothers have?")
    print(f"Reasoning: {reasoning}")
    print(f"Content: {content}")
    
    # Example multiple concurrent requests
    prompts = [
        "What is 2+2?",
        "What is 3+3?",
        "What is 4+4?"
    ]
    tasks = [get_completion(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    
    for prompt, (content, reasoning) in zip(prompts, results):
        print(f"\nPrompt: {prompt}")
        print(f"Reasoning: {reasoning}")
        print(f"Content: {content}")

# Run the async code
await main()