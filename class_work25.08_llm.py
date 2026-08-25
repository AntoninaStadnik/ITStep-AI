# Завдання 1
# Напишіть модель для рекомендації книг з двох ланцюгів:
#  Перший ланцюг отримує назву книги та визначає її жанр
#  Другий отримує назву книги, жанр та повертає список
# схожих книг(того ж самого жанру та іншого)

import dotenv
import os

import langchain
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)

# class BookGenre(BaseModel):
#     genre: str = Field(description="визначення жанру книги")
#
#
# parser = PydanticOutputParser(pydantic_object= BookGenre)
#
# instructions = parser.get_format_instructions()
# #print(instructions)
#
# prompt = PromptTemplate.from_template("""
#     Ти -- чат бот по визначення книг/бібліотекар
#     Задача: ти отримуєш назву книги, потрібно надати рекомендацію по книги і назвати жанр
#
#     ###ІНСТРУКЦІЇ###
#     1. відповідь має бути короткою (1 слово)
#     2. відповіді мають бу4ти літературною, красивою мовою
#
#     ###ФОРМАТ ВІДПОВІДІ###
#     {format_instruction}
#
#     ###ВХІДНІ ДАННІ###
#     {book_name}
#
# """,
#     partial_variables={
#         "format_instruction": instructions
#     }
#  )
# chain1 = prompt | llm | parser
#
# book = "Заклинатель"
#
# data = {
#     "book_name": book
# }
#
# response = chain1.invoke(data)
# print(response.genre)
#
#
# class BooksRecommendation(BaseModel):
#     book_name: list[str] = Field(description = "отримуєш назву книги, жанр та повертає список схожих книг і інші книги")
#
# parser = PydanticOutputParser(pydantic_object= BooksRecommendation)
#
# instructions = parser.get_format_instructions()
#
#
# prompt = PromptTemplate.from_template("""
#         Ти - книжковий експерт
#         Задача: навести приклади книг, які можуть бути такого ж жанру і можуть сподобатися читачу
#         і також запропонуй інші книги, інших жанрів, які можуть сподобатися читачу
#
#         ###ІНСТРУКЦІЇ###
#         1. до кожної рекомендації надай дуже короткий опис книги(1 речення)
#         2. надай не більше 4 рекомендацій
#
#         ###ВХІДНІ ДАНІ###
#         {book_name}
#         {genre}
#         {format_instructions}
# """,
#     partial_variables={
#         "format_instructions": instructions
#     }
# )
#
# chain2 = prompt | llm | parser
#
# book = "Заклинатель"
#
# data = {
#     "book_name": book
# }
#
# response1 = chain1.invoke(data)
#
# data = {
#     "book_name": response1.genre,
#     "genre": response1.genre
# }
#
# response2 = chain2.invoke(data)
#
# print(f"Список рекомендацій і жанри:")
#
# for book in response2.books:
#     print(book)


# Напишіть модель для генерації резюме:
#  Перший ланцюг отримує опис вакансії та повертає
# основні навички, які необхідні
#  Другий ланцюг отримує основні навички та опис
# кандидата і генерує резюме

class Skills(BaseModel):
    experience: float = Field(description = "Досвід робготи в роках")
    language_level: str = Field(description = "рівень англійської мови")
    frameworks: list[str] = Field(description = "кількість потрібних бібліотек")
    technologies: list[str] = Field(description = "список технологій, котрі треба знати")
    programming_language: list[str] = Field(description= "список мов які треба знати")

parser = PydanticOutputParser(pydantic_object= Skills)

instructions = parser.get_format_instructions()
#print(instructions)

prompt = PromptTemplate.from_template("""
        Ти -- досвідчений технічний рекрутер
        Задача: прочитай опис вакансії дістань основні навички
        
        ###ФОРМАТ ВІДПОВІДІ###
        {instructions}
        
        ###ВХІДНІ ДІННІ###
        Опис вакансії: {vacancy_description}
        
""",
        partial_variables={"instructions": instructions}
)

chain = prompt | llm | parser

vacancy_description = """
Are you a Data Scientist with a love of LLMs, generative AI?
 
We are looking for a passionate Data Scientist to implement AI solutions aimed at achieving business goals.
 
This role offers the opportunity to work on cutting-edge AI adoption projects that helps to improve current business processes.
 
     You'll be a great fit if you have:
 Strong Python Experience (2 year +);
Experience with LLM , Diffusion models;
Knowledge of Prompt engineering;
Experience with Gen AI-related technologies such as LangChain and RAG;
Experience with Neural Networks (Optional) ;
Experience with NLP , Predictive analytics and Machine learning;
Experience with Pandas;
Experience with SQL, including experience with large datasets;
Strong experience in statistics;
Bachelor's degree in Computer Science or a related field.
What you'll do:
Develop AI agents that utilize LLM, RAG and langchain approach;
Implement LLM and Diffusion models to boost business productivity;
Utilize LLM (LLM Vision) to improve object detection, text classification and extraction;
Create forecasting, recommendation, and classification models;
Transform business challenges to AI applications.

     We ensure your growth with:
Competitive salary fixed in USD;
Flexible working schedule and fully remote work format;
Paid vacation days and sick leave days ;
Personal and professional development opportunities;
Participation in building innovative projects from scratch using modern technologies;
Team-building activities and corporate events;
English classes and educational events.
"""

data = {
    "vacancy_description": vacancy_description
}

response = chain.invoke(data)
print(response)




