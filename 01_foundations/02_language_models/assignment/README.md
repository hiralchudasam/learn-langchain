# Assignment 02 — Language Models

## 🎯 Goal

Build a simple **multi-turn CLI chatbot** that uses a ChatModel with:
- A system prompt defining the bot's persona
- Streaming token output
- A conversation history that grows with each turn
- A way to exit cleanly

---

## Requirements

1. Use `ChatOpenAI` (or `ChatAnthropic` if you prefer)
2. Set a **system prompt** that gives the bot a personality (e.g., "You are a sarcastic but helpful assistant")
3. Accept user input in a loop
4. Print the bot's response while **streaming** (token by token)
5. Maintain conversation history across turns
6. Exit the loop when user types `quit` or `exit`
7. Print **total tokens used** at the end of the session

---

## Expected Output

```
🤖 Chatbot ready. Type 'exit' to quit.

You: Hello! What can you do?
Bot: [streams response...]

You: Remember that. My name is Rahul.
Bot: [streams response...]

You: What's my name?
Bot: [streams response, mentions Rahul...]

You: exit
👋 Goodbye! Total tokens used: 342
```

---

## Bonus Challenges

- [ ] Allow the user to switch between models mid-conversation (type `switch gpt-4o`)
- [ ] Show the number of tokens used per turn
- [ ] Save the conversation to a `.txt` file when the user exits
- [ ] Add a command to `clear` the conversation history

---

## Hints

- Use `llm.stream(messages)` and print each chunk
- Store history as a list of `HumanMessage` and `AIMessage` objects
- Access token counts from `response.usage_metadata`
- Wrap the loop in a `try/except KeyboardInterrupt` for Ctrl+C handling

---

## Reference Solution

See [`solution.py`](./solution.py) after attempting it yourself!
