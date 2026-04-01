"""
18 — LangSmith
Example 01: Auto-tracing, manual tracing, evaluation, token cost tracking
"""

from dotenv import load_dotenv
import os

load_dotenv()

# LangSmith auto-traces when these env vars are set (no code changes needed):
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_key
# LANGCHAIN_PROJECT=langchain-learning

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback
from langsmith import traceable, Client

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ─── 1. Auto-Tracing (zero code) ─────────────────────────────────────────────

print("=" * 55)
print("1. Auto-Tracing (all runs go to LangSmith automatically)")
print("=" * 55)

chain = (
    ChatPromptTemplate.from_template("What are 3 key facts about {topic}?")
    | llm
    | StrOutputParser()
)

result = chain.invoke({"topic": "LangSmith"})
print(result)
print("\n→ View this trace at: https://smith.langchain.com")

# ─── 2. @traceable — trace any Python function ───────────────────────────────

print("\n" + "=" * 55)
print("2. @traceable Decorator")
print("=" * 55)

@traceable(name="preprocess_query")
def preprocess(query: str) -> str:
    """Clean and normalize the user query."""
    return query.strip().lower().replace("  ", " ")

@traceable(name="postprocess_response")
def postprocess(response: str) -> str:
    """Format the response for display."""
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    return "\n".join(f"• {l}" if not l.startswith("•") else l for l in lines)

@traceable(name="full_pipeline")
def run_pipeline(user_input: str) -> str:
    clean_input = preprocess(user_input)
    raw_output = chain.invoke({"topic": clean_input})
    return postprocess(raw_output)

result = run_pipeline("  LangChain Framework  ")
print(result)

# ─── 3. Token Cost Tracking ───────────────────────────────────────────────────

print("\n" + "=" * 55)
print("3. Token Cost Tracking (get_openai_callback)")
print("=" * 55)

with get_openai_callback() as cb:
    result1 = chain.invoke({"topic": "vector databases"})
    result2 = chain.invoke({"topic": "RAG patterns"})

print(f"Total calls: 2")
print(f"Prompt tokens:     {cb.prompt_tokens}")
print(f"Completion tokens: {cb.completion_tokens}")
print(f"Total tokens:      {cb.total_tokens}")
print(f"Total cost (USD):  ${cb.total_cost:.4f}")

# ─── 4. LangSmith Client (Datasets & Evals) ──────────────────────────────────

print("\n" + "=" * 55)
print("4. LangSmith Datasets & Evaluation")
print("=" * 55)

if os.getenv("LANGCHAIN_API_KEY"):
    try:
        client = Client()

        # Create dataset
        dataset_name = "langchain-learning-qa-demo"
        if not client.has_dataset(dataset_name=dataset_name):
            dataset = client.create_dataset(dataset_name, description="Demo QA dataset")
            client.create_examples(
                inputs=[
                    {"topic": "LangChain"},
                    {"topic": "RAG"},
                    {"topic": "LangGraph"},
                ],
                outputs=[
                    {"answer": "LangChain is a framework for LLM applications"},
                    {"answer": "RAG combines retrieval with generation"},
                    {"answer": "LangGraph enables stateful agent graphs"},
                ],
                dataset_id=dataset.id,
            )
            print(f"Created dataset: '{dataset_name}' with 3 examples")
        else:
            print(f"Dataset '{dataset_name}' already exists")
            dataset = client.read_dataset(dataset_name=dataset_name)

        # Run evaluation
        from langsmith.evaluation import evaluate

        def predict(inputs: dict) -> dict:
            result = chain.invoke({"topic": inputs["topic"]})
            return {"output": result}

        experiment = evaluate(
            predict,
            data=dataset_name,
            experiment_prefix="gpt-4o-mini-facts",
            description="Testing facts chain",
        )
        print(f"Evaluation complete — view at smith.langchain.com")

    except Exception as e:
        print(f"LangSmith not configured or error: {e}")
        print("Set LANGCHAIN_API_KEY in .env to use LangSmith features")
else:
    print("LANGCHAIN_API_KEY not set — skipping LangSmith client demo")
    print("Add it to .env to enable full tracing and evaluation")

# ─── 5. Run Metadata & Tags ───────────────────────────────────────────────────

print("\n" + "=" * 55)
print("5. Adding Metadata and Tags to Runs")
print("=" * 55)

result = chain.invoke(
    {"topic": "embeddings"},
    config={
        "tags": ["production", "v2"],
        "metadata": {
            "user_id": "user_42",
            "session_id": "sess_abc123",
            "environment": "dev",
        },
        "run_name": "embeddings-facts-run",
    },
)
print(f"Result: {result[:100]}...")
print("→ Run tagged with metadata, visible in LangSmith UI")
