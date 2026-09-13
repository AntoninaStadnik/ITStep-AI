# Напишіть додаток з чат ботом по допомозі з вивченням
# англійської мови.
#  Якщо користувач просить перекласти слово або
# фразу, то вивести переклад та приклад використання
# у речені
#  Якщо користувач просить перекласти речення, то
# вивести переклад та пояснення граматики, наприклад
# структура there is/are, пасивна форма дієслова, тощо

import streamlit as st

# заголовок сайту
st.title("Чат бот з вивчення англійської мови")

# ЧАТ-БОТ

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    trim_messages,
)

# завантажити дані з .env
api_key = st.secrets["GEMINI_API_KEY"]

# модель
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",  # назва моделі
    api_key=api_key  # ключ до сервера з моделлю
)


# історія повідомлень
if "history" not in st.session_state:

    st.session_state.history = [
        SystemMessage(
            f"""
            Ти -- викладач з англійської мови.
            Твоя задача перекладати слова на англійську. Наводити приклад використання в реченні.
            Перекладай не дослівно, а так,, як би то звучало природньо українською.
            Якщо користувач вводить речення на переклад - вивести переклад і пояснити граматику. 
            Навести приклад перекладу речення у різних часах: present, pasr, future/
            """
        )
    ]


# отримати повідомлення від користувача
user_text = st.chat_input("Введіть повідомлення")

# якщо повідомлення не None тоді викликаємо чат бот
if user_text is not None:
    # створити HumanMessage
    human_message = HumanMessage(content=user_text)

    # отримати історію повідомлень
    messages = st.session_state.history

    # додати повідомлення в історію
    messages.append(human_message)

    # отримати відповідь моделі
    response = llm.invoke(messages)

    # додати response в історію спілкування
    messages.append(response)


# вивести всю історію повідомлень
for message in st.session_state.history:

    # не показувати SystemMessage
    if isinstance(message, SystemMessage):
        continue


    # отримуємо тип повідомлення
    role = ""

    if isinstance(message, HumanMessage):
        role = "user"
    else:
        role = "AI"

    text = message.content
    if isinstance(message, AIMessage):
        text = text[0]['text']

    with st.chat_message(role):
        st.markdown(text)