# Passive Journal AI Assistant

An AI-powered chatbot for Passive Journal using RAG (Retrieval Augmented Generation) with Streamlit.

## Setup

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

3. Run the application
```bash
python -m streamlit run rag.py
```

## Features
- Interactive chat interface
- Context-aware responses using RAG
- Integration with Passive Journal's content
- Real-time response generation

## Technologies Used
- Streamlit
- LangChain
- OpenAI
- FAISS
