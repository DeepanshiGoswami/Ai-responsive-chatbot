import os
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, add_messages, START, END
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("groq_api_key")

# Define chat state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize LLM
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)


# Define chatbot node
def chatbot(state: State):
    return {"messages": llm.invoke(state["messages"])}

# Set up LangGraph
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
my_chatbot = graph.compile()

# Streamlit settings
st.set_page_config(page_title="Groq Chatbot", layout="centered")

# --- Stylish Chat UI ---
st.markdown("""
    <style>
    .user-bubble {
        background-color: #007bff;
        color: white;
        padding: 12px 16px;
        border-radius: 16px;
        margin: 8px 0;
        max-width: 80%;
        align-self: flex-end;
        text-align: right;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .bot-bubble {
        background: linear-gradient(to right, #8e2de2, #4a00e0);
        color: black;
        padding: 12px 16px;
        border-radius: 16px;
        margin: 8px 0;
        max-width: 80%;
        align-self: flex-start;
        text-align: left;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🤖 Chatbot Assistant</h1>", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_input" not in st.session_state:
    st.session_state.last_input = ""

# Display chat history
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for role, msg in st.session_state.messages:
    if role == "user":
        st.markdown(f"<div class='user-bubble'>🙋‍♀️ You: {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>🤖 Assistant: {msg}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Input field
user_input = st.text_input("You:", "", key="user_input")

# Process input
if user_input and user_input != st.session_state.last_input:
    st.session_state.messages.append(("user", user_input))
    state = {"messages": st.session_state.messages}
    response = my_chatbot.invoke(state)
    reply = response["messages"][-1].content
    st.session_state.messages.append(("assistant", reply))
    st.session_state.last_input = user_input
    st.rerun()

# Action buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_input = ""
        st.rerun()

with col2:
    if st.button("❌ Exit Chat"):
        st.markdown("Chatbot: Goodbye! 👋")
