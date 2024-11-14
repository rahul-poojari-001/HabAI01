import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables
dotenv.load_dotenv()

gemini_api_key = st.secrets["API_KEY"]
if gemini_api_key is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=gemini_api_key)

# Functions for PDF processing and question answering
def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(raw_text)

def get_vector(chunks):
    if not chunks:
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=gemini_api_key)
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def user_question(question, db, chain, raw_text):
    if db is None:
        return "Please upload and process a PDF first."

    docs = db.similarity_search(question, k=5)
    response = chain.invoke(
        {"input_documents": docs, "question": question, "context": raw_text},
        return_only_outputs=True
    )
    return response.get("output_text")


def conversation_chain(wake_up_time, sleep_time):
    # Crafting a template to help the AI generate personalized study plans or routines.
    instructions = f"""
    You are a productivity assistant designed to help students create and maintain an effective study routine. 
    Based on the student's wake-up time and sleep time, generate a personalized study plan for the day. 
    Consider the following factors for the study plan:
    - Wake-up time: {wake_up_time}
    - Sleep time: {sleep_time}
    - Include study sessions, breaks, and productivity tips.
    - Add motivational elements to keep the student engaged throughout the study day.
    - Provide study times for various subjects if provided or based on the student's preferences.
    - Suggest strategies to stay focused and manage distractions.
    
    """
    context = """ 
     Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    template = instructions + context
    
    model_instance = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=gemini_api_key)
    prompt = PromptTemplate(template=template, input_variables=["wake_up_time", "sleep_time", "context", "question"])
    return load_qa_chain(model_instance, chain_type="stuff", prompt=prompt), model_instance

def main():
    # Set the page configuration
    st.set_page_config(page_title="HabAI - Gives Productivity 👨‍🏫", page_icon="👨‍🏫", layout="wide")
    st.header("HabAI - Gives Productivity 👨‍🏫")

    # Initialize session state variables specific to this page
    if "vector_store_chatbot_2" not in st.session_state:
        st.session_state.vector_store_chatbot_2 = None
    if "chain_chatbot_2" not in st.session_state:
        st.session_state.chain_chatbot_2 = None
    if "raw_text_chatbot_2" not in st.session_state:
        st.session_state.raw_text_chatbot_2 = None

    wake_up_time = st.time_input("Wake up time", value=None,key = "wake_up_time")
    sleep_time = st.time_input("Sleep", value=None, key = "sleep_time")

    pdf_docs = "ilovepdf_merged.pdf" #kb for routine design in pdf format 
    raw_text = get_pdf_text(pdf_docs)
    chunks = get_text_chunks(raw_text)
    vector_store = get_vector(chunks)
    chain, _ = conversation_chain(wake_up_time, sleep_time)
    
    if vector_store and chain and raw_text:
        st.session_state.vector_store_chatbot_2 = vector_store
        st.session_state.chain_chatbot_2 = chain
        st.session_state.raw_text_chatbot_2 = raw_text

        user_query = f"I wake up at {wake_up_time} and go to sleep at {sleep_time}. Can you suggest a daily routine that optimizes my time for productivity, rest, and self-care? Include specific times for meals, exercise, work, breaks, and relaxation activities."
                            # Initial question for disease identification
        if wake_up_time and sleep_time:
            response = user_question(user_query, st.session_state.vector_store_chatbot_2, st.session_state.chain_chatbot_2, st.session_state.raw_text_chatbot_2)
            st.write(response)
            
if __name__ == "__main__":
    main()