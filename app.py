import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Session state for preprocessed data
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = False

st.set_page_config(page_title="Content Engine Streaming Bot", page_icon="🤖")
st.title("Content Engine Streaming Bot 🤖")
st.subheader("Developed By: Abhijit Mandal")

# Load user query
user_query = st.chat_input("Your Message")

# Load API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
PINECONE_API_INDEX = os.environ.get('PINECONE_API_INDEX')
HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

# Function to preprocess data
def preprocess_data():
    with st.spinner("Loading data and initializing models..."):

        # Load PDFs
        loader = PyPDFDirectoryLoader(path='pdfs/', glob="**/*.pdf")
        pdfs = loader.load()
        st.write("PDF data loaded successfully!")

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        docs = splitter.split_documents(pdfs)
        st.write("Document Splitting created successfully!")

        # Load embedding model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.write("Embeddings model loaded successfully!")

        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
        index_name = "medical-chatbot"
        namespace = "Default"

        # Create vector database with Pinecone
        docsearch = PineconeVectorStore.from_documents(
            documents=docs,
            index_name=index_name,
            embedding=embeddings, 
            namespace=namespace
        )
        st.write("Pinecone created the Vector DB successfully")

        # Load LLM
        model_path = "openai-community/gpt2"
        llm = HuggingFaceHub(
            repo_id=model_path, 
            model_kwargs={'temperature': 0.5, 'max_length': 200}
        )
        st.write("LLM loaded successfully")

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k":2}))
        st.write("RAG Pipeline created")

        st.session_state.qa = qa
        st.session_state.llm = llm
        st.session_state.preprocessed = True

# Run preprocessing if not already done
if not st.session_state.preprocessed:
    preprocess_data()

qa = st.session_state.qa
llm = st.session_state.llm

def get_response(query, chat_history):
    template = """
        You're a helpful assistant. Answer the following questions considering the history of the conversation.
        Chat History: {chat_history}
        User Question: {user_question}
    """

    chat_history_text = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])

    prompt = template.format(chat_history=chat_history_text, user_question=query)
    
    result = qa.invoke(query)
    answer = result['result'].split('Helpful Answer: ')[-1].strip()

    return answer

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# Chat interface
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query, role="user"))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = get_response(user_query, st.session_state.chat_history)
        st.markdown(ai_response)

    st.session_state.chat_history.append(AIMessage(content=ai_response, role="assistant"))