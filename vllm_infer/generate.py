# vllm_infer/generate.py
from openai import OpenAI

def get_weather_answer(prompt):
    client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")
    response = client.chat.completions.create(
        model="weather_lora",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512
    )
    return response.choices[0].message.content
