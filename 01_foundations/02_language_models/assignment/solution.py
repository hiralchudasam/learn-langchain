"""
Assignment 02 — Language Models
Solution: Streaming CLI chatbot with conversation history and token tracking
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()


def run_chatbot(
    model: str = "gpt-4o-mini",
    system_prompt: str = "You are a helpful, friendly assistant.",
    temperature: float = 0.7,
):
    llm = ChatOpenAI(model=model, temperature=temperature, streaming=True)

    history = [SystemMessage(content=system_prompt)]
    total_input_tokens = 0
    total_output_tokens = 0

    print(f"\n🤖 Chatbot ready (model: {model}). Type 'exit' to quit.\n")

    try:
        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                break

            if user_input.lower() == "clear":
                history = [SystemMessage(content=system_prompt)]
                print("🗑️  Conversation cleared.\n")
                continue

            if user_input.lower().startswith("switch "):
                new_model = user_input.split(" ", 1)[1].strip()
                llm = ChatOpenAI(model=new_model, temperature=temperature, streaming=True)
                model = new_model
                print(f"🔀 Switched to model: {model}\n")
                continue

            history.append(HumanMessage(content=user_input))

            print("Bot: ", end="", flush=True)
            full_response = ""

            # Stream token by token
            for chunk in llm.stream(history):
                content = chunk.content
                print(content, end="", flush=True)
                full_response += content

            print()  # newline after streaming ends

            # Try to get token counts (available on final chunk in some versions)
            try:
                response = llm.invoke(history[-1:])  # just for metadata
                usage = response.usage_metadata
                if usage:
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)
            except Exception:
                pass

            history.append(AIMessage(content=full_response))
            print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted.")

    total = total_input_tokens + total_output_tokens
    print(f"\n👋 Goodbye! Session stats:")
    print(f"   Input tokens:  {total_input_tokens}")
    print(f"   Output tokens: {total_output_tokens}")
    print(f"   Total tokens:  {total}")


if __name__ == "__main__":
    run_chatbot(
        model="gpt-4o-mini",
        system_prompt="You are a sarcastic but helpful assistant who speaks with wit.",
        temperature=0.8,
    )
