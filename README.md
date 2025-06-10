# Document Chat with Gemini and InLegalBERT

A Streamlit application that allows you to chat with your documents using Google's Gemini AI. This application implements a RAG (Retrieval Augmented Generation) model to provide accurate answers based on your document content.

## Features

- PDF document processing with OCR support
- Legal document analysis using InLegalBERT
- Chat interface powered by Google Gemini
- Support for multiple document formats
- Document embeddings and semantic search

## Prerequisites

- Python 3.9 or higher
- Google Cloud Vision API credentials
- Google Gemini API key
- Streamlit account for deployment

## Setup

1. Clone the repository:
```bash
git clone https://github.com/godspeed-003/law-mini.git
cd law-mini
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/vision_key.json
```

## Local Development

Run the Streamlit app locally:
```bash
streamlit run app.py
```

## Deployment

1. Create a Streamlit account at https://streamlit.io
2. Connect your GitHub repository
3. Add your environment variables in Streamlit Cloud:
   - GEMINI_API_KEY
   - GOOGLE_APPLICATION_CREDENTIALS (content of your Vision API key JSON)
4. Deploy your app

## Architecture

- `app.py`: Main Streamlit application
- `utils.py`: Core functionality for document processing and AI models
- `.env`: Environment variables (not in version control)
- `requirements.txt`: Project dependencies

## Supported File Types

### Text Documents
- Text files (.txt)
- PDF files (.pdf)
- Microsoft Word documents (.docx)
- HTML files (.html)

### Spreadsheets
- CSV files (.csv)
- Excel files (.xls, .xlsx)

### Data Files
- JSON files (.json)
- XML files (.xml)
- YAML files (.yml, .yaml)

### Presentations
- PowerPoint files (.pptx)

### Additional Features
- Fallback to text reading for unknown file types
- Structured data extraction from various formats
- Preserved formatting where applicable

## How It Works

1. **Document Processing**:
   - The application processes your documents and splits them into manageable chunks
   - Each file type is handled with its specific parser to extract text content
   - Documents are processed in batches to manage token limits

2. **Embedding Generation**:
   - Creates embeddings for each chunk using Google's Generative AI
   - Uses the embedding-001 model for efficient vector representation

3. **Vector Storage**:
   - The chunks are stored in a FAISS vector database for efficient retrieval
   - FAISS enables fast similarity search across document chunks

4. **Question Answering**:
   - When you ask a question, the system:
     - Retrieves the most relevant document chunks
     - Uses Gemini AI to generate a response based on the retrieved context
   - The RAG model ensures responses are grounded in your document content

## Security

- API key is stored securely in environment variables
- All processing is done locally
- Documents are processed in temporary storage
- The `.env` file should be added to `.gitignore` to prevent accidental commits
- No document content is stored permanently

## Performance Considerations

- Documents are processed in chunks to manage token limits
- Text splitting is optimized for context preservation
- Vector search is efficient using FAISS
- Batch processing for multiple documents

## Troubleshooting

If you encounter any issues:

1. **API Key Issues**:
   - Ensure your API key is correctly set in the `.env` file
   - Verify the API key has the necessary permissions

2. **File Processing Issues**:
   - Check if the file type is supported
   - Ensure the file is not corrupted
   - Verify file encoding (UTF-8 recommended)

3. **Model Errors**:
   - Ensure you're using a compatible version of the Gemini API
   - Check your internet connection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Google Gemini AI
- InLegalBERT
- Streamlit