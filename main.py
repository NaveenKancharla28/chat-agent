import os
from langchain_openai import ChatOpenAI


from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("✅ Loaded Key:", api_key[:5] + "..." if api_key else " No Key Found")


# Create LLM (language model)
llm = ChatOpenAI(api_key=api_key, temperature=0.7)




# Add memory to keep past chat
memory = ConversationBufferMemory()

# Build a conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = conversation.predict(input=user_input)
    print("AI:", response)
