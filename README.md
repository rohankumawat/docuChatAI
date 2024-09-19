# DocuChatAI

"DocuChatAI" is a simple and intuitive web app that allows users to interact with their PDF documents using natural language processing. Built using LangChain and deployed on Streamlit Cloud, the app extracts information from documents, enabling users to ask questions about their content and receive instant answers.

This app will be continuously updated to include features such as reranking of document content, advanced retrieval-augmented generation (RAG) techniques, and support for loading web pages for analysis.

## Features
- **Basic PDF Question-Answering:** Upload a PDF, ask questions, and get responses based on the content of the document.
- **Fast and Responsive UI:** Streamlit provides a seamless interface for quick interactions with your uploaded documents.
- **LangChain Integration:** Powered by LangChain, the app efficiently parses and retrieves relevant information from the uploaded document.

## Upcoming Features
- **Document Reranking:** Improve the accuracy and relevance of answers by reranking sections of the document.
- **Advanced Retrieval-Augmented Generation (RAG):** Further improve the ability to retrieve and synthesize responses from multiple documents.
- **Web Page Loaders:** Add the capability to load and interact with web pages, broadening the scope of the app beyond PDFs.
- **Question-Answer Chat History:** Implement chat history so users can track their previous interactions for better context.

## How to Use
1. **Upload a PDF:** Choose a PDF file from your local system to upload into the app.
2. **Ask Questions:** Use the chat interface to ask questions about the content of the document. The app will process your query and provide an answer based on the uploaded document.
3. **Explore Responses:** Get instant responses based on the context of the document using LangChain's retrieval model.

## Installation (For Local Development)
To run the app locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/rohankumawat/docuChatAI.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deployment
The app is deployed on [Streamlit Cloud](https://streamlit.io/), making it easily accessible without local installation. Visit [this link](https://docuchatlangchain.streamlit.app) to start chatting with your documents!

## Technologies Used
- **LangChain**: For processing and retrieving relevant sections of documents. [Learn more](https://python.langchain.com/docs/)
- **Streamlit**: For building the web interface and deploying the app. [Learn more](https://docs.streamlit.io/)
- **Python**: The core programming language for building this app.
  
## Future Roadmap
- [ ] Integrate document reranking to improve answer accuracy.
- [ ] Implement advanced Retrieval-Augmented Generation (RAG).
- [ ] Add support for other document formats and web page loading.
- [ ] Improve scalability and optimize performance for larger documents.

## Useful Resources
- [LangChain PDF QA Documentation](https://python.langchain.com/docs/tutorials/pdf_qa/)
- [LangChain QA with Chat History](https://python.langchain.com/docs/tutorials/qa_chat_history/)
- [Streamlit LLM Quickstart](https://docs.streamlit.io/develop/tutorials/llms/llm-quickstart)

## Contributions
Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.