# 🧠 AI-Powered Document QA ChatBot

## Live View
Check out the live demo of this AI-powered document-based question-answering chatbot: [Live Demo](https://ai-powered-question-answering-chatbot.streamlit.app/).

---

## Problem
In today's data-driven world, businesses and individuals are constantly dealing with large amounts of documents in various formats (PDFs, CSVs, Excel, JSON). Extracting meaningful insights or answering specific questions from these documents can be a time-consuming and challenging task, especially without a smart AI assistant.

The problem lies in:
- How to quickly extract relevant information from vast amounts of data spread across different document formats.
- Providing a seamless, intuitive way to query documents and get precise answers.
- Enabling efficient document-based question answering without relying on manual document review.

---

## Solution
Introducing the **AI-Powered Document QA ChatBot**: A Retrieval-Augmented Generation (RAG) based solution that leverages cutting-edge Groq LLMs, FAISS vector databases, and Hugging Face embeddings to empower users to ask questions and instantly retrieve answers from their uploaded documents.

This chatbot supports multiple document formats like PDF, CSV, Excel, and JSON, and allows users to interact with their data in a natural, intuitive way. Whether you're analyzing financial reports, academic papers, or any other structured data, this chatbot can extract insights and provide answers with speed and accuracy.

---

## Features
- **Multi-Format Document Support**: Upload and process documents in PDF, CSV, Excel, and JSON formats.
- **Retrieval-Augmented Generation (RAG)**: Uses Groq's advanced LLMs for efficient document retrieval and question answering.
- **Hugging Face Embeddings**: Utilize Hugging Face embeddings to convert document text into vectors for fast similarity search.
- **Custom Model & API Key Selection**: Choose between different Groq models (Llama 3.3, DeepSeek R1 and Gemma) based on your needs and input your own API key.
- **Document Chunking & Querying**: Automatically chunks large documents into manageable parts and retrieves relevant information based on user queries.
- **Real-Time Query Responses**: Ask questions from the uploaded document and get answers in real-time.
- **Embedded Context Display**: See the top-k relevant chunks of the document retrieved to support the answer.

---

## Tech Stacks
- **Python**: The backend language for processing and managing data.
- **Streamlit**: For building the interactive web interface.
- **LangChain**: For chaining document loaders, embeddings, and retrieval logic.
- **FAISS**: To store and retrieve embeddings efficiently in vector databases.
- **Groq LLMs**: For generating accurate and context-based answers from the documents.
- **HuggingFace Transformers**: To create embeddings for the documents.
- **dotenv**: To manage environment variables securely, like the API key for Groq.

---

## Summary
The **AI-Powered Document QA ChatBot** is designed to simplify document-based question answering. By utilizing Groq's powerful LLMs and Hugging Face embeddings in combination with FAISS for vector-based document retrieval, this solution offers users a seamless experience to query their documents and retrieve meaningful answers instantly.

This project aims to streamline the process of document analysis by providing a smart chatbot interface that can easily handle a wide range of document formats. Whether you're working with PDFs, spreadsheets, or structured JSON data, this solution provides the necessary tools to get precise answers in a fraction of the time compared to traditional methods.

---

## Installation

To run the project locally, follow these steps:

1. Clone the repository:
  ```
  git clone https://github.com/shibbir-ahmad24/AI-Powered-Document-QA-ChatBot.git
  ```
2. Navigate into the project directory:
  ```
  cd ai-powered-document-qa-chatbot
  ```
3. Create a virtual environment:
  ```
  python -m venv venv
  ```
4. Activate the virtual environment:
  - On Windows:
    ```
    .\venv\Scripts\activate
    ```
  - On macOS/Linux:
    ```
    source venv/bin/activate
    ```
5. Install the required dependencies:
  ```
  pip install -r requirements.txt
  ```
6. Set up your environment variables:
  - Create a .env file in the root of the project and add your Groq API Key:
    ```
      GROQ_API_KEY=your_groq_api_key_here
    ```
7. Run the app:
  ```
  streamlit run app.py
  ```

## APP UI

![q1](https://github.com/shibbir-ahmad24/AI-Powered-Document-QA-ChatBot/blob/main/Figures/q1.png)

![q2](https://github.com/shibbir-ahmad24/AI-Powered-Document-QA-ChatBot/blob/main/Figures/q2.png)

![q3](https://github.com/shibbir-ahmad24/AI-Powered-Document-QA-ChatBot/blob/main/Figures/q3.png)

![q4](https://github.com/shibbir-ahmad24/AI-Powered-Document-QA-ChatBot/blob/main/Figures/q4.png)

