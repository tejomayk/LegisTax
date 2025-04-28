# LegisTax - AI-Powered Research Assistant for International Tax Law
### Overview
LegisTax is an AI-powered research assistant designed to help accountants efficiently navigate large collections of international tax law documents. By leveraging advanced natural language processing, retrieval-augmented generation (RAG), and hybrid search techniques, LegisTax streamlines legal research, saving time and improving accuracy.

### Features
1. AI-Powered Search: Quickly retrieve relevant international tax law information using a combination of keyword search and vector embeddings.
2. Document Processing Pipeline: Extracts and preprocesses text from large PDF collections using PyMuPDF.
3. Retrieval: Combines FAISS for semantic search and BM25 for keyword-based retrieval.
4. User-Friendly Interface: An interactive web-based search tool built with Streamlit.

### Contributing
I welcome contributions! Please:

- Fork the repository.
- Create a feature branch.
- Submit a pull request with detailed explanations.

### License
This project is licensed under the MIT License

### How to run
Simply install dependencies by running the following command:
```
> pip install -r requirements.txt
```
Then, run the following command to start the application locally:
```
> streamlit run app.py 
```