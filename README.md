Chat with PDF using LangChain

###Overview
This project enables users to interact with PDF documents using natural language. By leveraging the LangChain library, it facilitates seamless extraction and querying of PDF content through a chatbot interface. This solution can be particularly useful for quickly retrieving specific information from extensive documents, such as reports, manuals, or academic papers.

###Features

-PDF Upload: Users can upload PDF documents to the application.
-Content Extraction: Extract text content from the uploaded PDF.
-Natural Language Query: Use natural language to query the content of the PDF.
-Chat Interface: Interact with the PDF content through a chat interface.
-Result Display: Display query results in a user-friendly format.

1) Clone the repository:
     -git clone https://github.com/mahnoorakhtar9015/chat-with-pdf-langchain.git
     -cd chat-with-pdf-langchain

2) Create a virtual environment:
   -python -m venv venv
   -source venv/bin/activate

3) Install dependencies:
    -pip install -r requirements.txt
    

4) Set up your OpenAI API Key:
    -OPENAI_API_KEY=your_openai_api_key_here

###USAGE
1) Run the application:
       -streamlit run app.py
   
2) Upload a PDF:
   -Open the Streamlit app in your browser.
   -Use the upload button to upload your PDF document.
   
3) Interact with the PDF:
   -Type your queries in the chat interface.
   -The application will extract relevant content from the PDF and display the results.


