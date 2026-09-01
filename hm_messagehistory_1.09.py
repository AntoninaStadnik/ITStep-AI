# Напишіть чат модель яка підсумовує всю розмову в
# декілька речень. Вкажіть щоб модель зберігала якомога
# більше деталей.
# Використайте цю модель для простого чат бота який
# замість trim_massages використовує модель з підсумуванням.
# Підсумовуйте повідомлення, коли їх більше 4.
# Старі повідомлення треба видалити
# НЕ ВИДАЛЯТИ SystemMessage та не використовувати
# його для підсумування

import dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)


messages: list[BaseMessage] = [
    SystemMessage("""
    Ти -- ввічливий чатбот
    Твоя задача підтримувати спілкування з користувачем

    ### ІНСТРУКЦІЇ ###
    1. Відповіді мають бути короткими (до 2 речень)
    2. Зберігай як можна більше деталей в ході розмови
    3. Запам'ятовуй факти про користувача
    """),
]


while True:
    user_text = input("Ви: ")

    if user_text == "":
        break

    human_message = HumanMessage(content=user_text)

    messages.append(human_message)

    count = 0

    for message in messages:
        if not isinstance(message, SystemMessage):
            count += 1

    if count > 4:

        text = ""

        for message in messages:
            if not isinstance(message, SystemMessage):
                text += message.content
                text += "\n"

        summary_messages = [
            SystemMessage("""
            Ти -- модель для підсумування розмови.

            Підсумуй всю розмову в декілька речень.
            Зберігай якомога більше деталей:
            факти про користувача, його вподобання,
            питання, відповіді та важливий контекст.

            Не вигадуй нову інформацію.
            """),

            HumanMessage(content=text)
        ]

        summary = llm.invoke(summary_messages)

        new_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                new_messages.append(message)

        messages = new_messages
        messages.append(
            AIMessage(
                content=f"Підсумок попередньої розмови: {summary.content}"
            )
        )


    response = llm.invoke(messages)


    print(f"AI: {response.content}")

    messages.append(response)

    # вивести історію
    print()
    print("----------------------------------")
    print("HISTORY")

    for message in messages:
        print(message)

    print("----------------------------------")
    print()


