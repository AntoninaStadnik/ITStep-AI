# Напишіть чат бота, який спілкується у стилі різних
# персонажів книг\фільмів або відомих людей. Ким саме бути
# чат бот вирішує з повідомлення від користувача.
# Якщо персонаж або книга невідомі, то відповісти що
# невідома інформація та запропонувати декілька відомих
# прикладів на вибір

import dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage, trim_messages,
    BaseMessage
)
from pydantic import BaseModel
from english_parser import chain

# завантадити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# # модель
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",   # назва моделі
    api_key=api_key    # ключ до сервера з моделлю
)

# message = [
#
#     SystemMessage(""""
#     Ти -- чатбот, який імітує спілкування різних персонажів з0 кних, фільмів, казок.
#     Твоя задача - надавати відповіді користувачу у відповідному стилі в залежності від персонажа
#
#     ###ІНСТРУКЦІЇ###
#     1. Відповіді надавай чітко і коротко(1-2 речення)
#     2. Стиль відповіді вибирай залежно від запиту користувача.
#     3. Якщо персонаж або книга невідомі, то відповісти що невідома інформація та запропонувати декілька відомих прикладів на вибір
#     """)
# ]
#
# while True:
#     user_text = input("User: ")
#
#     if user_text == "":
#         break
#
#     user_message = HumanMessage(content=user_text)
#
#     message.append(user_message)
#
#     response = llm.invoke(message)
#
#     print(f"AI: {response.content}")
#
#     message.append(response)

# Напишіть чат бота, який дає відповіді на питання
# стосовно умов повернення товару.
# Якщо користувач запитує щось інше, то відповідати що
# немає інформації.
# Застосуйте обмеження історії(можна десь 5 повідомлень)

# with open("C:/Users/GamePC/Desktop/python lessons/AI/ITStep-AI/data/lesson9/return_policy.txt", "r", encoding="utf-8") as f:
#     rules = f.read()
#
# message = [
#     SystemMessage(f"""
#     Ти -- консультант магазину
#     Твоя задача давати відповіді на питання стосовно умов повернення товару
#
#     ###ІНСТРУКЦІЇ###
#     1. Відповіді мають бути короткими(до 2 речень)
#     2. Якщо користувач запитує щось інше, відповідай, що немає інформації
#     3. Нічого не вигадуй, відповідай чітко за правилами повернення, якщо в правилах
#        не вистачає інформації пиши, що не знаєш.
#
#     ###ПРАВИЛА ПОВЕРНЕННЯ###
#     {rules}
#     """)
# ]
#
# trimmer = trim_messages(
#     strategy='last',  # залишати останні повідомлення
#
#     token_counter=len,  # рахуємо кількість повідомлень
#     max_tokens=5,  # залишати максимум 5 повідомлення(System, AI, Human)
#
#     start_on='human',  # історія завжди починатиметься з HumanMessage
#     end_on='human',  # історія завжди закінчуватиметься з HumanMessage
#     include_system=True  # SystemMessage не чіпати
# )
#
# while True:
#     user_text = input("You: ")
#
#     if user_text == "":
#         break
#
#     human_message = HumanMessage(content=user_text)
#
#     message.append(human_message)
#
#     message = trimmer.invoke(message)
#
#     response = llm.invoke(message)
#
#     print(f"AI: {response.text}")
#
#     message.append(response)

# Напишіть чат бота, який допомагає у вивченні
# англійської мови з наступним функціоналом:
#  якщо користувач просить перекласти слово або фразу
# то дається переклад слова та приклад використання в
# реченні
#  якщо користувач просить перекласти речення, то
# Практичне завдання
# дається переклад самого речення, а також пояснення
# граматики, наприклад структура there is\are, питання в
# різних часових формах, тощо.
# Приклади реалізуйте як HumanMessage та AIMessage

messages: list[BaseMessage] = [
    SystemMessage("""
    Ти -- досвідчений викладач англійської мови
    Твоя задача допомагати вивчати англійську мову

    ###ІНСТРУКЦІЇ###
    1. Відповіді мають бути короткими(до 2 речень)
    2. Якщо користувач запитує щось інше, відповідай, що немає інформації
    3. Якщо користувач просить надати переклад слова, перекладай і наводь приклад в реченні і різні форми: теперішній, минулий, майбутній
    4. Поясню граматику з прикладами
    """),

    HumanMessage("Переклади слово - яблуко"),
    AIMessage("an apple, apple - fruit and it's very healthy food")
]

while True:
    user_text = input("You: ")

    if user_text == "":
        break

    human_message = HumanMessage(user_text)

    messages.append(human_message)

    response = llm.invoke(messages)

    data = {
        "text": response.text
    }

    eng_words = chain.invoke(data)

    print(f"AI: {response.content}")
    print(eng_words)

    messages.append(response)


# Модифікуйте попереднє завдання таким чином, щоб в
# SystemMessage передавався список вивчених слів
# користувачем.
# Для цього напишіть окрему модель яка буде діставати з
# відповіді(AIMessage) усі англійські слова(вважаємо що
# користувач знає лише ті слова, про які йому сказала модель).
# Список вивчених слів треба зберігати в json файлі та
# відвантажувати при запуску програми.
# Змініть функціонал таким чином:
#  якщо користувач просить перекласти слово або фразу
# то дається переклад слова та приклад використання в
# реченні з вивченими словами
#  якщо користувач просить перекласти речення, то
# додатково пояснюється значення невідомих слів


