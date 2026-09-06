# Напишіть чат бота, з інструментом по рекомендації
# ресторанів.
# Для цього скористайтесь
# GoogleSerperAPIWrapper(type="places")
# Інструмент повинен отримувати запит для пошуку та
# повертати таку інформацію про ресторани:
#  назва
#  посилання на сайт(якщо є)
#  рейтинг

import dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

# завантадити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# # модель
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",   # назва моделі
    api_key=api_key    # ключ до сервера з моделлю
)

google_places = GoogleSerperAPIWrapper(
    type="places"
)

@tool
def search_restaurant(query: str) -> str:
    """
    Search restaurant
    :param query: str - user query
    :return:
    """

    print("hi from search restaurant")
    result = google_places.results(query)

    return result


agent = create_agent(
    model = llm,
    tools=[search_restaurant]
)

messages = [
    SystemMessage(
        """
        Ти -- ресторанний чавт - бот
        
        У тебе є доступ до інструмента
        * search_restaurant
        
        ###ІНСТРУКЦІЯ###
        1. Надавай короткі відповіді, не більше 5 назв ресторанів
        2. Якщо користувач пропустив інформацію, назву міста, пиши що недостаньо інформації
        3. Виводь інформацію у наступному виді: назва ресторану, посилання на сайт(якщо є), рейтинг
        """
    )
]

while True:
    user_input = input("You: ")

    if user_input == "":
        break

    user_message = HumanMessage(content=user_input)

    messages.append(user_message)

    data = {
        "messages": messages
    }

    data = agent.invoke(data)

    messages = data["messages"]

    # відповідь моделі -- останнє повідомлення в історії
    response = messages[-1]

    # вивести відповідь на екран
    print(response.text)

    # виведення історії
    print()
    print("----------ІСТОРІЯ-----------")

    for message in messages:
        print(repr(message))  # вивести разом з назсою класу

    print("-----------------------------")
    print()



