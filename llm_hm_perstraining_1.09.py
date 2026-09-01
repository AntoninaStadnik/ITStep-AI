# Напишіть модель для генерації персонального плану
# тренувань з двох ланцюгів:
#  Перший ланцюг отримує мету тренування(схуднення,
# набір м’язів, тощо) та повертає список вправ
#  Другий ланцюг отримує список вправ, рівень
# підготовки користувача(низький, середній,
# професіонал) та кількість часу на тиждень(в годинах)
# і повертає план тренувань

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


class TrainingPlan(BaseModel):
    exercises: list[str] = Field(description="список вправ")


parser = PydanticOutputParser(pydantic_object=TrainingPlan)

instructions = parser.get_format_instructions()

prompt1 = PromptTemplate.from_template("""
    Ти -- досвідчений тренер, досвід 5+ років
    Задача: виходячи з потреб відвідувача(схуднення, набір м’язів, тощо), запропонувати список безпечних вправ,
    врахуй момент, що деякі вправи можуть впливати на колінні чашечки, спину.

    ###ІНСТРУКЦІЇ###
    1. кожна відповідь має бути у форматі: назва вправи - техніка(не більше 2х речень)

    ###ВХІДНІ ДАНІ###
    Мета: {training}

    Відповідь:

    ### ФОРМАТ ВІДПОВІДІ ###
    {format_instructions}
""",
partial_variables={"format_instructions": instructions}
)

chain1 = prompt1 | llm | parser

user_goal = "схуднути"

data = {
    "training": user_goal
}


class ExerciseList(BaseModel):
    week_training: list[str] = Field(description="план тренувань на тиждень")


parser2 = PydanticOutputParser(pydantic_object=ExerciseList)

instructions2 = parser2.get_format_instructions()

prompt = PromptTemplate.from_template("""
    Ти -- досвідчений фітнес тренер.
    Твоя задача сформувати тижневий план тренувань для клієнта, враховуючи рівень підготовки(низький, середній, високий),
    а також кількість часу в залі(годин на тиждень)

    ###ІНСТРУКЦІЇ###
    1. план має бути у форматі: вправа - техніка
    2. врахуй травматичність вправ, добав застереження по техніці
    3. наголоси, що будь-яка мета досягається комплексно спорт + харчування

    ###ВХІДНІ ДАНІ###
    Вправи: {exercises}
    Рівень підготовки: {level}
    Час на тиждень: {hours} годин

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

""",
    partial_variables={"format_instructions": instructions2}
)

chain2 = prompt | llm | parser2

user_goal = "схуднути"

data = {
    "training": user_goal
}

response1 = chain1.invoke(data)

print(f"Відповідь на питання: {response1.exercises}")

data = {
    "exercises": response1.exercises,
    "level": "низький",
    "hours": 4
}

response2 = chain2.invoke(data)

print(f"План на тиждень:")
for plan in response2.week_training:
    print(plan)