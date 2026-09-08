# Напишіть додаток, який симулює спілкування з певною
# відомою людиною.
# З ким саме спілкуватись вводить користувач через
# st.text_input()

import streamlit as st

# заголовок сайту
st.title("Спілкування з видатною особою")

# з ким саме спілкуватися
person = st.text_input("З ким ви хочете поспілкуватися?")

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
            Ти -- {person}.
            Твоя задача симулювати спілкування з користувачем
            так, ніби ти ця відома людина.
            Відповідай у стилі та відповідно до поглядів {person}.
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

    with st.chat_message(role):
        st.markdown(message.content)