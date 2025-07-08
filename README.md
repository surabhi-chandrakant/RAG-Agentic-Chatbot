# RAG Agentic Chatbot

An intelligent Retrieval-Augmented Generation (RAG) system built with agentic workflow architecture. This chatbot can process multiple document types, create semantic embeddings, and provide intelligent responses based on your uploaded documents.

## 🌐 Live Demo

**Try the app live**: [https://huggingface.co/spaces/surabhic/Agentic_RAG_Chatbot](https://huggingface.co/spaces/surabhic/Agentic_RAG_Chatbot)

Upload your documents and start chatting with them instantly!

## 🚀 Features

- **Multi-format Document Support**: PDF, DOCX, PPTX, CSV, TXT, and Markdown files
- **Agentic Architecture**: Modular agent-based system using Model Context Protocol (MCP)
- **Semantic Search**: FAISS-powered vector search with sentence transformers
- **LLM Integration**: Google Gemini AI for intelligent response generation
- **Interactive Chat Interface**: Streamlit-based web UI
- **Real-time Processing**: Asynchronous document processing and query handling

## 🏗️ Architecture

The system consists of four main agents:

1. **Ingestion Agent**: Processes and chunks uploaded documents
2. **Retrieval Agent**: Creates embeddings and performs semantic search
3. **LLM Response Agent**: Generates intelligent responses using retrieved context
4. **Coordinator Agent**: Orchestrates the workflow between agents

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- Google API Key (for Gemini AI)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/surabhi-chandrakant/RAG-Agentic-Chatbot.git
   cd RAG-Agentic-Chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```
   
   Or set the environment variable directly:
   ```bash
   export GOOGLE_API_KEY="your_google_api_key_here"
   ```

5. **Get Google API Key**
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key and add it to your environment variables

## 📦 Dependencies

Create a `requirements.txt` file with the following dependencies:

```txt
streamlit>=1.28.0
PyPDF2>=3.0.1
python-docx>=0.8.11
python-pptx>=0.6.21
pandas>=1.5.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
google-generativeai>=0.3.0
numpy>=1.24.0
asyncio-throttle>=1.0.2
python-dotenv>=1.0.0
```

## 🚀 Usage

### Running Locally

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the interface**
   - Open your browser and navigate to `http://localhost:8501`

3. **Upload documents**
   - Use the file uploader to add PDF, DOCX, PPTX, CSV, or TXT files
   - Click "Process" to index the document

4. **Start chatting**
   - Type your questions in the chat interface
   - The system will search through your documents and provide intelligent responses

### Deployment on Hugging Face Spaces

1. **Prepare your repository**
   ```bash
   # Add necessary files
   touch app.py
   touch requirements.txt
   touch README.md
   ```

2. **Create Hugging Face Space**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Streamlit" as the SDK
   - Connect your GitHub repository

3. **Configure secrets**
   - In your Hugging Face Space settings, add:
     - `GOOGLE_API_KEY`: Your Google API key

4. **Deploy**
   - Push your code to the connected repository
   - The space will automatically build and deploy

## 🎯 Example Use Cases

- **Research Assistant**: Upload research papers and ask questions about methodologies, findings, or specific topics
- **Document Analysis**: Process business documents, contracts, or reports for quick insights
- **Educational Tool**: Upload textbooks or course materials for interactive learning
- **Technical Documentation**: Query API docs, manuals, or technical specifications

## 📁 Project Structure

```
RAG-Agentic-Chatbot/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .env                  # Environment variables (not in repo)
├── .gitignore           # Git ignore file
└── docs/                # Additional documentation
```

## 🔍 How It Works

1. **Document Ingestion**: Upload files are processed by the Ingestion Agent, which extracts text and splits it into chunks
2. **Embedding Creation**: The Retrieval Agent creates vector embeddings using sentence transformers
3. **Indexing**: Embeddings are stored in a FAISS index for fast similarity search
4. **Query Processing**: User queries are embedded and matched against the document index
5. **Response Generation**: Retrieved context is sent to Google Gemini for intelligent response generation

## 🛠️ Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Required for Google Gemini API access

### Customization Options

- **Chunk Size**: Modify `chunk_size` in `IngestionAgent.chunk_text()` (default: 250 words)
- **Embedding Model**: Change the sentence transformer model in `RetrievalAgent.__init__()`
- **LLM Model**: Switch between different Gemini models in `LLMResponseAgent.__init__()`
- **Retrieval Count**: Adjust `top_k` parameter for number of chunks retrieved

## 🐛 Troubleshooting

### Common Issues

1. **Import Error for google-generativeai**
   ```bash
   pip install google-generativeai
   ```

2. **FAISS Installation Issues**
   ```bash
   pip install faiss-cpu  # For CPU-only version
   # or
   pip install faiss-gpu  # For GPU version
   ```

3. **Streamlit Port Issues**
   ```bash
   streamlit run app.py --server.port 8502
   ```

4. **Memory Issues with Large Documents**
   - Reduce chunk size or increase overlap
   - Process documents in smaller batches

### API Key Issues

- Ensure your Google API key is valid and has proper permissions
- Check that the environment variable is correctly set
- Verify API quota limits haven't been exceeded

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request



## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web interface
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [Google Gemini](https://ai.google.dev/) for language model capabilities

## 📞 Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/surabhi-chandrakant/RAG-Agentic-Chatbot/issues)
- Check the troubleshooting section above

---

**Note**: This application processes documents locally and sends only relevant chunks to the LLM for response generation. Your documents are not stored permanently and are only used for the duration of your session.
