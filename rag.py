import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Function to initialize the RAG components
@st.cache_resource
def initialize_rag():
    # Remove hardcoded API key and use environment variable instead
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Get from .env file
    
    # Use JSONLoader instead of TextLoader
    loader = JSONLoader(
        file_path="passive_journal_content.json",
        jq_schema='.[]',  # This extracts each item in the top-level array
        text_content=False
    )
    documents = loader.load()
    
    # No need for text splitting if each JSON object is already a suitable size
    # If splitting is still needed, uncomment the following lines:
    # text_split = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    # documents = text_split.split_documents(documents)
    
    db = FAISS.from_documents(documents, OpenAIEmbeddings())
    
    llm = ChatOpenAI(model_name="gpt-4o-mini")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI agent of Passive Journal. Your job is to talk with visitors of Passive Journal and give 
        information about Passive Journal and its services.
         Passive Journal is a platform for passive income ideas. It has multiple courses on affiliate marketing, dropshipping, digital marketing, Facebook Ads,
         Instagram Marketing, SEO, YouTube, Video editing, and more. 
         Passive Journal also have a newsletter called Firday, They consult students to study in abroad.
         
        Try to fit in the customer to a user persona wheather thea are student who wants to study abroad or someone who wants to learn digital marketing to recommend products. 
        You task is to recommend products to the user based on their persona.
        It is extremely important to ask user questions to get more information about user's requirements and work pattern if you do not have enough 
        information to make an informed decision. It is absolutely mandatory to recommend only the Passive Journal courses and services.
        Passive Journal has one time payment courses. If user pay one time he will get lifetime access to all of Passive Journal the courses and services. Context:

    <context>
    {context}
    </context>
             """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    history_aware_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])
    
    retriever = db.as_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, history_aware_prompt)
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# Streamlit UI
st.title("Passive Journal AI Assistant")

# Add clickable logo
st.markdown(
    """
    <div style="background-color: white; padding: 10px; display: inline-block; border-radius: 5px;">
        f'<a href="https://passivejournal.com/">'
        <img src="https://passivejournal.com/wp-content/uploads/2024/08/logo-b.png" width="200">
        f'</a>',
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize RAG components
rag_chain = initialize_rag()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask about Passive Journal...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Convert chat history to the format expected by the RAG chain
        rag_history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in st.session_state.chat_history[:-1]  # Exclude the latest user message
        ]
        
        # Generate response
        ai_response = rag_chain.invoke({"input": user_input, "chat_history": rag_history})
        full_response = ai_response["answer"]
        
        message_placeholder.markdown(full_response)
    
    # Add AI response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
