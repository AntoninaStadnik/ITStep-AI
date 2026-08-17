# Прочитайте файл data\lesson9\return_policy.txt Та
# напишіть простий чат бот для відповідей на питання
# користувачів стосовно повернення товару. Діалог завершується
# коли користувач вводить порожній рядок.
# Передавайте усю історію спілкування у форматі:
# Instruction: ….
# Human: massage1
# AI: message2
# Human: massage3
# AI: message4
# Human: massage5
# AI:

import os
import dotenv
from langchain_google_genai import GoogleGenerativeAI

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    temperature=1.2,
    top_k=10
)

with open("data/lesson9/return_policy.txt", encoding="utf-8") as file:
    rules = file.read()

instructions = f"""
    Ти консультант магазину.

    Ось правила повернення товару:
    {rules}

    Відповідай на питання клієнта тільки на основі цих правил.
    Якщо відповіді на питання немає в правилах, відповідай:
    "Я не знаю відповіді на це питання."

    Відповідь має бути короткою — не більше 2 речень.
    """

history = []

while True:
    question = input("How can I help you: ")

    if not question:
        break

    history.append(f"Human: {question}")

    prompt = instructions + "\n" + "\n".join(history) + "\nAI:"

    response = llm.invoke(prompt)

    print(f"AI: {response}")

    history.append(f"AI: {response}")
