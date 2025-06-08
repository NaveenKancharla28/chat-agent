import os
import io
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.agents import Tool, initialize_agent
from langchain.tools import DuckDuckGoSearchRun
# from pydub import AudioSegment   # keep only if audio conversion is needed
# import speech_recognition as sr  # keep only if you use voice later

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Page setup
st.set_page_config(page_title="Naveen's GPT", page_icon="🤖")
st.title("🤖 Naveen's GPT")

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.pop("conversation", None)
    st.session_state.pop("chat_log", None)
    st.rerun()

# Setup agent + memory if not already
if "conversation" not in st.session_state:
    embedding_model = OpenAIEmbeddings(api_key=api_key)
    vectorstore = Chroma(
        collection_name="chat_memory",
        embedding_function=embedding_model,
        persist_directory="vectorstore"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    memory = VectorStoreRetrieverMemory(retriever=retriever)

    search_tool = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="WebSearch",
            func=search_tool.run,
            description="Useful for searching current events or general topics online."
        )
    ]

    llm = ChatOpenAI(api_key=api_key, temperature=0.7)
    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True,
        memory=memory
    )

    st.session_state.conversation = agent
    st.session_state.chat_log = []

# 🔤 Text-only Chat
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.chat_log.append(("user", user_input))
    with st.chat_message("user", avatar="🧍‍♂️"):
        st.write(user_input)

    response = st.session_state.conversation.run(user_input)
    st.session_state.chat_log.append(("ai", response))
    with st.chat_message("assistant", avatar="🧠"):
        st.write(response)

# Chat log section
st.divider()
st.subheader("📜 Entire Chat Log")
for role, msg in st.session_state.chat_log:
    if role == "user":
        st.markdown(f"🧍‍♂️ **You:** {msg}")
    else:
        st.markdown(f"🧠 **AI:** {msg}")
