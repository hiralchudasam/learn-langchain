"""
04 — Output Parsers
Example 01: StrOutputParser, JsonOutputParser, PydanticOutputParser, structured output
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ─── 1. StrOutputParser ───────────────────────────────────────────────────────

chain = (
    ChatPromptTemplate.from_template("What is the capital of {country}?")
    | llm
    | StrOutputParser()
)
print("StrOutputParser:", chain.invoke({"country": "India"}))

# ─── 2. CommaSeparatedListOutputParser ────────────────────────────────────────

list_parser = CommaSeparatedListOutputParser()

list_chain = (
    ChatPromptTemplate.from_template(
        "List 5 programming languages. {format_instructions}"
    ).partial(format_instructions=list_parser.get_format_instructions())
    | llm
    | list_parser
)
languages = list_chain.invoke({})
print("\nCommaSeparatedList:", languages)
print("Type:", type(languages))   # → list

# ─── 3. PydanticOutputParser ──────────────────────────────────────────────────

class Movie(BaseModel):
    title: str = Field(description="The movie title")
    year: int = Field(description="Release year")
    genre: str = Field(description="Main genre")
    rating: float = Field(description="IMDb rating out of 10")
    director: str = Field(description="Director's name")

parser = PydanticOutputParser(pydantic_object=Movie)

movie_template = ChatPromptTemplate.from_messages([
    ("system", "You are a movie database. Always respond in the requested format.\n{format_instructions}"),
    ("human", "Give me info about the movie: {movie_name}"),
]).partial(format_instructions=parser.get_format_instructions())

movie_chain = movie_template | llm | parser

movie = movie_chain.invoke({"movie_name": "Inception"})
print(f"\nPydanticOutputParser:")
print(f"  Title:    {movie.title}")
print(f"  Year:     {movie.year}")
print(f"  Genre:    {movie.genre}")
print(f"  Rating:   {movie.rating}")
print(f"  Director: {movie.director}")

# ─── 4. JsonOutputParser ──────────────────────────────────────────────────────

json_parser = JsonOutputParser()

json_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Return a JSON object with keys: name, age, city."),
        ("human", "Create a fake person profile."),
    ])
    | llm
    | json_parser
)

profile = json_chain.invoke({})
print("\nJsonOutputParser:", profile)

# ─── 5. Structured Output (Modern Approach) ───────────────────────────────────

class Person(BaseModel):
    name: str
    job: str
    fun_fact: str

structured_llm = llm.with_structured_output(Person)
person = structured_llm.invoke("Tell me about a fictional scientist")
print(f"\nStructured output → {person.name}, {person.job}: {person.fun_fact}")

# ─── 6. List of Structured Objects ───────────────────────────────────────────

class BookList(BaseModel):
    books: List[Movie]

structured_list_llm = llm.with_structured_output(BookList)
# Demonstrates nested Pydantic models work too
