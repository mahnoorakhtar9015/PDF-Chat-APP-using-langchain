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
    st.header('Chat with PDF')
   
    # upload pdf

    pdf=st.file_uploader("Upload your pdf",type='pdf')
    

    if pdf is not None:
        pdf_reader=PyPDF2.PdfReader(pdf)
        text=""
        # Extracting text from Pdf
        for page in pdf_reader.pages:
             text+=page.extract_text()
    
        text_splitter=RecursiveCharacterTextSplitter(
             chunk_size=1000,
             chunk_overlap=200,
             length_function=len
        )
         # Split text into chunks
        chunks = text_splitter.split_text(text=text)

        # Initialize OpenAI embeddings with model and API key
        embedding = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=OPENAI_API_KEY)

        # Convert chunks into embeddings using FAISS
        Vector_store = FAISS.from_texts(chunks, embedding=embedding)
      
        
        
                
        question = st.text_input("Question:")


        if st.button("Send"):

                if question:

                    context=Vector_store.similarity_search(query=question, k=3)
                        

                    prompt_template = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""
                        Given the following context:
                        {context}

                        Please answer the following question:
                        {question}
                        """
                        )

                    llm = OpenAI(api_key=OPENAI_API_KEY)
                    chain = LLMChain(llm=llm, prompt=prompt_template)

                    response = chain.run({"context": context, "question": question})
            
                    st.markdown("#### Answer:")
                    st.success(response)

                  
    else:
        st.warning("Please upload a PDF file.")
if __name__=='__main__':
      main()



