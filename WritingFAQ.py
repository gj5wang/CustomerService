import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

SYSTEM_PROMPT = '''
You are helping build a customer support FAQ knowledge base for a smart grill product.

Given one FAQ item, generate:
1. similar_questions: different ways a real customer may ask the same question.
2. keywords: short search keywords useful for retrieval.

Rules:
- Do not change the original question or answer.
- Do not invent product features or policies.
- Keep the same meaning as the FAQ.
- similar_questions should be natural customer wording.
- keywords should be concise and useful for search.
- Output valid JSON only.

Output format:
{
  "similar_questions": [],
  "keywords": []
}
'''

completion = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Category:After-sales Support\nQuestion:What does the warranty cover?\nAnswer:We offer a 3-year limited warranty covering manufacturing defects under normal residential use. Warranty does NOT cover normal wear and tear, cosmetic damage, rust, misuse, or grease fires."}
    ]
)

print(completion.choices[0].message.content)