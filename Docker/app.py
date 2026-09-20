from openai import OpenAI
import os

client = OpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key='docker'
)
completion = client.chat.completions.create(
    model="ai/llama3.2:1B-Q4_0",
    messages = [
        {"role": "system", "content": "Answer the question in a couple sentences."},
        {"role": "user", "content": "about LLaMA 3.2 model"}
    ]
)

print(completion.choices[0].message.content)