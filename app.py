import streamlit as st
import PyPDF2
import os
import openai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain



# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')



# Configure OpenAI with the API key
openai.api_key = OPENAI_API_KEY
# Add CSS for sidebar color
# Add CSS for sidebar color
# Define custom CSS for sidebar and entire app
# Define custom CSS for sidebar and entire app

st.set_page_config(page_title="PDF Chat", page_icon="📄", layout="wide")

# Content inside the sidebar
st.sidebar.title('LLM Chat App')
st.sidebar.markdown('''
## About
This app is an LLM-powered chatbot built using:
- Streamlit
- Langchain
- OpenAI
''')
st.sidebar.write('Made by Mahnoor')


def main():
   
    # Header
    st.title("📄 Chat with Your PDF")
    st.write("Upload a PDF and ask questions about its content.")

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PyPDF2.PdfReader(pdf)
        text = ""
        
        # Extracting text from PDF
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Split text into chunks
        chunks = text_splitter.split_text(text=text)

        # Initialize OpenAI embeddings with model and API key
        embedding = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=OPENAI_API_KEY)

        # Convert chunks into embeddings using FAISS
        vector_store = FAISS.from_texts(chunks, embedding=embedding)
        
        # Chat interface
        st.subheader("Chat with your PDF")

        # Initialize session state for chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask a question about the PDF"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Typing..."):

                    # Get context for the question
                    context = vector_store.similarity_search(query=prompt, k=3)

                    # Create prompt template
                    prompt_template = PromptTemplate(
                        input_variables=["context", "question"],
                        template="""
                        Given the following context:
                        {context}

                        Please answer the following question:
                        {question}

                        Instructions:
                        - If the question is related to greetings response with greetings.
                        """
                    )

                    # Initialize OpenAI LLM
                    llm = OpenAI(api_key=OPENAI_API_KEY)
                    chain = LLMChain(llm=llm, prompt=prompt_template)

                    # Generate response
                    response = chain.run({"context": context, "question": prompt})

                # Display assistant response in chat message container
                
                    st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload a PDF file to begin.")
if __name__=='__main__':
      main()



