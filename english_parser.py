import dotenv
import os

import langchain
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# завантадити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# # модель
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",   # назва моделі
    api_key=api_key    # ключ до сервера з моделлю
)


class Response(BaseModel):
    words: list[str] =Field(description="список унікальних слів")

parser = PydanticOutputParser(pydantic_object=Response)

instructions = parser.get_format_instructions()

prompt = PromptTemplate.from_template("""
    Твоя задача дістати всі унікальні фнглійські слова з тексту.
    
    ###ІНСТРУКЦІЇ###
    1. Ігноруй артиклі
    2. Ігноруй стоп словва(am, is, are, he, she, it, тощо)
    
    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}   
    
     ###ВХІДНІ ДАННІ###  
     {text}
""",

    partial_variables={"format_instructions": instructions}

)

chain = prompt | llm | parser

if __name__ == "__main__":

    text = """
    Ось кілька прикладів слова book у різних значеннях:
    I am reading a book. — Я читаю книгу.
    She bought a new book yesterday. — Вона вчора купила нову книгу.
    I want to book a hotel room. — Я хочу забронювати номер у готелі.
    We booked a flight to London. — Ми забронювали рейс до Лондона.
    Тобто book може бути іменником «книга» або дієсловом «бронювати».
    """

    data = {
        "text": text
    }

    response = chain.invoke(data)
    print(response)