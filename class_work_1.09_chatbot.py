# Напишіть функцію яка перевіряє складність паролю:
#  кількість символів(>8)
#  наявність хоча б однієї літери\цифри\спеціального
# символу
#  наявність літер в різних регістрах
# Функція повертає тест з описом паролю(що добре, а що
# погано)
# На основі цієї функції створіть агента.


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
    BaseMessage,
    trim_messages,
)

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
serper_key = os.getenv("SERPER_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=api_key
)

serper_search = GoogleSerperAPIWrapper(
    serper_api_key=serper_key
)

def count_char(password: str):
     total_alpha = 0
     total_digit = 0
     total_special = 0
     total_upper = 0
     total_lower = 0
     for char in password:
         if char.isalpha():
             total_alpha += 1
         elif char.isdigit():
             total_digit += 1
         else:
             total_special += 1
         if char.isupper():
             total_upper += 1
         if char.islower():
             total_lower += 1
     return total_alpha, total_digit, total_special, total_upper, total_lower

@tool
def search_person(name: str)-> str:
    """
    Person search, looking for person in internet
    :param name: str -- person name
    :return:
    """

    info = serper_search.results(f"News about {name}")
    return info



@tool
def password_check(password: str):
     """
     Password verification
     :param password: str -- password
     :return:
     """

     if len(password) <= 8:
         return "Password should have at least 8 characters"

     total_alpha, total_digit, total_special, total_upper, total_lower= count_char(password)

     if total_alpha == 0:
             return "Password should have letter"
     if total_digit == 0:
             return "Password should have digit"
     if total_upper == 0:
             return "Password should have uppercase letter"
     if total_lower == 0:
             return "Password should have lowercase letter"
     if total_special == 0:
             return "Password should have special character"
     else:
             return "Password is good"

agent = create_agent(
     model=llm,
     tools=[password_check, search_person],
 )

messages = [
     SystemMessage(
         """
         Ти -- чат бот

         """
     )
 ]

while True:
     user_pass = input("You: ")

     if user_pass == "":
         break

     user_message = HumanMessage(content=user_pass)

     messages.append(user_message)

     data = {
         "messages": messages
     }

     data = agent.invoke(data)

     messages = data["messages"]

     response = messages[-1]

     print(response.content)

# Напишіть модель показує останні новини про певну
# людину. Якщо користувач вводить не ім’я людини, то вивести
# повідомлення «немає відповідної інформації»
# Скористайтесь DuckDuckGoSearchRun






