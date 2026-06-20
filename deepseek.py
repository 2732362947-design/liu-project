from openai import OpenAI

client = OpenAI(
    api_key="sk-fed933a489ed4a6690002e1e3b26e1b3",
    base_url="https://api.deepseek.com"
)

def generate(prompt):
    res = client.chat.completions.create(
        model="deepseek-chat",   # 或 deepseek-chat / deepseek-reasoner
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content