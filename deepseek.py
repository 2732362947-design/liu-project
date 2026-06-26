from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def generate(prompt: str) -> str:
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("your_"):
        return f"[mock deepseek] Received prompt with {len(prompt)} characters."

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        return f"[deepseek error] {type(exc).__name__}: {exc}"
