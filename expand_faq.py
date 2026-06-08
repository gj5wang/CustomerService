import os
import json
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
- similar_questions should be natural customer wording.
- keywords should be concise and useful for search.
- Output valid JSON only. No extra text, no markdown backticks.

Output format:
{
  "similar_questions": [],
  "keywords": []
}
'''

# read the existing faq.json
with open("faq.json", "r") as f:
    faq = json.load(f)

# loop through every entry and fill in missing similar_questions and keywords
for entry in faq:
    if entry["similar_questions"] == [] or entry["keywords"] == []:
        print(f"Generating for: {entry['question']}")

        user_input = f"Category:{entry['category']}\nQuestion:{entry['question']}\nAnswer:{entry['answer']}"

        completion = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]
        )

        result = completion.choices[0].message.content
        parsed = json.loads(result)

        entry["similar_questions"] = parsed["similar_questions"]
        entry["keywords"] = parsed["keywords"]

        print(f"Done: {entry['question']}")

# save the updated faq.json
with open("faq.json", "w") as f:
    json.dump(faq, f, indent=4)

print("\nAll done. faq.json has been updated.")