"""
╔══════════════════════════════════════════════════════════════╗
║  Topic 04 — Output Parsers                                   ║
║  File:   02_structured_output.py                             ║
║  Level:  Advanced                                            ║
║  Goal:   Nested Pydantic models, enums, lists, error         ║
║          recovery, and choosing the right parser             ║
╚══════════════════════════════════════════════════════════════╝
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("=" * 60)
print("04 — Structured Output: Advanced Patterns")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# SECTION 1: Enums in Pydantic models
# Constrain values to a fixed set — the LLM can only pick valid options.
# ─────────────────────────────────────────────────────────────
print("\n📌 1. Enums — Constrained Values")
print("─" * 40)

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL  = "neutral"

class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"

class QuizQuestion(BaseModel):
    question:   str        = Field(description="The quiz question")
    answer:     str        = Field(description="The correct answer")
    difficulty: Difficulty = Field(description="Difficulty level: easy, medium, or hard")
    topic:      str        = Field(description="Topic category")

parser = PydanticOutputParser(pydantic_object=QuizQuestion)

template = ChatPromptTemplate.from_messages([
    ("system", "Generate a quiz question. {format_instructions}"),
    ("human", "Topic: {topic}, Difficulty: {difficulty}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = template | llm | parser

q = chain.invoke({"topic": "Python", "difficulty": "medium"})
print(f"  Question   : {q.question}")
print(f"  Answer     : {q.answer}")
print(f"  Difficulty : {q.difficulty.value}  ← Enum value, type-safe!")
print(f"  Topic      : {q.topic}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Nested Pydantic models
# Model complex, hierarchical data structures.
# ─────────────────────────────────────────────────────────────
print("\n📌 2. Nested Pydantic Models")
print("─" * 40)

class Author(BaseModel):
    name:        str = Field(description="Author's full name")
    nationality: str = Field(description="Author's country")

class Book(BaseModel):
    title:       str    = Field(description="Book title")
    author:      Author = Field(description="Book author details")
    year:        int    = Field(description="Publication year")
    genre:       str    = Field(description="Book genre")
    rating:      float  = Field(description="Rating out of 5", ge=0, le=5)
    description: str    = Field(description="One-sentence description")

book_parser = PydanticOutputParser(pydantic_object=Book)

book_template = ChatPromptTemplate.from_messages([
    ("system", "You are a book database. {format_instructions}"),
    ("human", "Give me details about: {book_title}"),
]).partial(format_instructions=book_parser.get_format_instructions())

book = (book_template | llm | book_parser).invoke({"book_title": "The Alchemist"})

print(f"  Title       : {book.title}")
print(f"  Author      : {book.author.name} ({book.author.nationality})")
print(f"  Year        : {book.year}")
print(f"  Rating      : {book.rating}/5")
print(f"  Type        : {type(book.author).__name__}  ← nested Pydantic!")

# ─────────────────────────────────────────────────────────────
# SECTION 3: List of objects
# Ask the LLM to return multiple items as a typed list.
# ─────────────────────────────────────────────────────────────
print("\n📌 3. List of Pydantic Objects")
print("─" * 40)

class Skill(BaseModel):
    name:        str = Field(description="Skill name")
    level:       str = Field(description="Beginner / Intermediate / Advanced")
    years:       int = Field(description="Years of experience")

class DeveloperProfile(BaseModel):
    name:   str         = Field(description="Developer name")
    role:   str         = Field(description="Job title")
    skills: List[Skill] = Field(description="List of technical skills")

profile_parser = PydanticOutputParser(pydantic_object=DeveloperProfile)

profile_template = ChatPromptTemplate.from_messages([
    ("system", "Generate a realistic developer profile. {format_instructions}"),
    ("human", "Create a profile for a {role} with {years} years experience."),
]).partial(format_instructions=profile_parser.get_format_instructions())

profile = (profile_template | llm | profile_parser).invoke(
    {"role": "Full Stack Developer", "years": "5"}
)

print(f"  Name  : {profile.name}")
print(f"  Role  : {profile.role}")
print(f"  Skills ({len(profile.skills)}):")
for skill in profile.skills[:4]:
    print(f"    • {skill.name:<20} {skill.level:<15} {skill.years}yr")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Custom Validators
# Add Python-level validation on top of Pydantic's type checking.
# ─────────────────────────────────────────────────────────────
print("\n📌 4. Custom Validators")
print("─" * 40)

class Product(BaseModel):
    name:     str   = Field(description="Product name")
    price:    float = Field(description="Price in USD", gt=0)
    category: str   = Field(description="Category")
    in_stock: bool  = Field(description="Is it available?")

    @field_validator("name")
    @classmethod
    def name_must_be_title_case(cls, v):
        return v.title()  # auto-normalize: "apple watch" → "Apple Watch"

    @field_validator("price")
    @classmethod
    def round_to_cents(cls, v):
        return round(v, 2)

product_parser = PydanticOutputParser(pydantic_object=Product)

product_template = ChatPromptTemplate.from_messages([
    ("system", "Generate a product. {format_instructions}"),
    ("human", "Create a {category} product."),
]).partial(format_instructions=product_parser.get_format_instructions())

product = (product_template | llm | product_parser).invoke({"category": "electronics"})
print(f"  Name     : {product.name}  ← auto title-cased")
print(f"  Price    : ${product.price}  ← rounded to cents")
print(f"  Category : {product.category}")
print(f"  In Stock : {product.in_stock}")

# ─────────────────────────────────────────────────────────────
# SECTION 5: OutputFixingParser — auto-retry on bad JSON
# If the LLM returns malformed JSON, this parser asks it to fix it.
# ─────────────────────────────────────────────────────────────
print("\n📌 5. OutputFixingParser — Resilient Parsing")
print("─" * 40)

fixing_parser = OutputFixingParser.from_llm(
    parser=product_parser,
    llm=llm,
)

# Simulate what happens when LLM returns bad JSON
bad_output = '{"name": "Laptop", "price": "1299.99", "category": "Electronics" }'
# ↑ Missing "in_stock" field and price is a string not float

try:
    result = fixing_parser.parse(bad_output)
    print(f"  Fixed output: {result.name}, ${result.price}, in_stock={result.in_stock}")
    print(f"  OutputFixingParser auto-corrected the malformed JSON!")
except Exception as e:
    print(f"  Fixing failed: {e}")

# ─────────────────────────────────────────────────────────────
# SECTION 6: Choosing the right parser — decision guide
# ─────────────────────────────────────────────────────────────
print("\n📌 6. Parser Selection Guide")
print("─" * 40)

guide = [
    ("StrOutputParser",            "Plain text response (most common)"),
    ("JsonOutputParser",           "Simple dict/JSON, no schema needed"),
    ("PydanticOutputParser",       "Typed object with validation"),
    ("CommaSeparatedListParser",   "Simple list of strings"),
    ("with_structured_output()",   "Modern: best for OpenAI models"),
    ("OutputFixingParser",         "Wrap any parser for auto-retry"),
]

for parser_name, use_case in guide:
    print(f"  {parser_name:<35} → {use_case}")

# ─────────────────────────────────────────────────────────────
# ⚠️  COMMON MISTAKE 1: Not injecting format_instructions
# The LLM won't know what format to use unless you tell it.
# Always: template.partial(format_instructions=parser.get_format_instructions())

# ⚠️  COMMON MISTAKE 2: Using PydanticOutputParser for simple output
# If you just want a string → use StrOutputParser
# If you want a dict → use JsonOutputParser
# Only use PydanticOutputParser when you need type safety + validation

# ⚠️  COMMON MISTAKE 3: temperature > 0 with structured parsers
# Higher temperature increases the chance of invalid JSON output.
# Always use temperature=0 with PydanticOutputParser.
# ─────────────────────────────────────────────────────────────
print("\n⚠️  Parser tips:")
print("   • Use temperature=0 with structured parsers")
print("   • Always inject format_instructions into your prompt")
print("   • Prefer with_structured_output() for OpenAI models")
print("   • Wrap with OutputFixingParser in production")
