# RAG VOICE ASSISTANT 🎙️

*Transforming Document Interactions with Voice Intelligence*

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red)
![OpenAI](https://img.shields.io/badge/OpenAI-API-orange)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-purple)

**Built with cutting-edge AI technologies:**

![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/-OpenAI-412991?style=flat-square&logo=openai&logoColor=white)
![LangChain](https://img.shields.io/badge/-LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/-FAISS-0084FF?style=flat-square&logo=meta&logoColor=white)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Backend](#running-the-backend)
  - [Running the Frontend](#running-the-frontend)
  - [Using the Application](#using-the-application)
- [API Documentation](#api-documentation)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

RAG Voice Assistant is an advanced AI-powered application that combines **Retrieval-Augmented Generation (RAG)** with **voice interaction capabilities**. The system allows users to upload PDF documents and interact with them through both text and voice interfaces, providing intelligent responses based on document content.

### Key Capabilities

🔍 **Document Intelligence**: Upload and process PDF documents with advanced text chunking and embedding  
🎙️ **Voice Interaction**: Real-time speech-to-text and text-to-speech capabilities  
💬 **Intelligent Chat**: Context-aware responses using OpenAI's GPT models  
🔊 **Audio Processing**: Support for multiple voice models and audio formats  
⚡ **Real-time Processing**: Live audio transcription and instant responses

---

## Features

### 🎯 Core Features
- **PDF Document Processing**: Advanced text extraction and chunking using PyPDF
- **Vector Search**: FAISS-powered similarity search for relevant document retrieval
- **Multi-modal Interaction**: Text, voice, and audio file input support
- **Real-time Transcription**: Live speech-to-text using OpenAI Whisper
- **Text-to-Speech**: Multiple voice options with OpenAI TTS
- **Context-aware Responses**: RAG-based intelligent document querying

### 🛠️ Technical Features
- **FastAPI Backend**: High-performance async API with automatic documentation
- **Streamlit Frontend**: Interactive web interface with multiple tabs
- **WebRTC Integration**: Real-time audio streaming capabilities
- **Modular Architecture**: Separate backend and frontend for scalability
- **Error Handling**: Comprehensive logging and error management
- **File Management**: Automatic cleanup and temporary file handling

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language**: Python 3.8+
- **Package Manager**: pip
- **API Keys**: OpenAI API key (required)
- **Audio Support**: System audio drivers for voice features

### Installation

Build the RAG Voice Assistant from source and install dependencies:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/rag-voice-assistant.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd rag-voice-assistant
   ```

3. **Install the dependencies:**
   
   Using **pip**:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. **Verify installation:**
   ```bash
   python -c "import openai; print('OpenAI installed successfully')"
   ```

---

## Usage

### Running the Backend

Start the FastAPI server with hot reload:

```bash
uvicorn main:app --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Running the Frontend

Launch the Streamlit interface:

```bash
streamlit run frontend.py
```

The web interface will open at: http://localhost:8501

### Using the Application

#### 1. Document Upload
- Navigate to the sidebar "Gerenciamento de Documentos"
- Upload one or more PDF files
- Click "Processar" to index the documents

#### 2. Chat Interface
- Use the "Chat" tab for text-based questions
- Ask questions about your uploaded documents
- Receive both text and audio responses

#### 3. Voice Input
- Switch to "Entrada por Voz" tab
- Grant microphone permissions
- Speak your questions naturally
- View real-time transcription

#### 4. Audio Features
- **Text-to-Speech**: Convert any text to audio with voice selection
- **Audio-to-Text**: Upload MP3 files for transcription
- **Voice Models**: Choose from 6 different voice options

---

## API Documentation

### Core Endpoints

#### Document Management
```http
POST /upload
Content-Type: multipart/form-data

Upload and process PDF documents for indexing.
```

#### Query Processing
```http
POST /query
Content-Type: application/json

{
  "question": "Your question about the documents"
}
```

#### Audio Processing
```http
POST /text-to-audio
Content-Type: multipart/form-data

Convert text to speech with voice selection.
```

```http
POST /audio-to-text
Content-Type: multipart/form-data

Transcribe audio files to text using Whisper.
```

### Response Formats

**Query Response:**
```json
{
  "response": "AI-generated answer based on document content"
}
```

**Error Response:**
```json
{
  "detail": "Error description"
}
```

---

## Architecture

### System Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    FastAPI      │    │    OpenAI       │
│   Frontend      │◄──►│    Backend      │◄──►│    Services     │
│                 │    │                 │    │                 │
│ • Chat UI       │    │ • RAG Pipeline  │    │ • GPT Models    │
│ • Voice Input   │    │ • Audio Proc.   │    │ • Whisper       │
│ • File Upload   │    │ • Vector Store  │    │ • TTS           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │    FAISS        │
                       │  Vector Store   │
                       │                 │
                       │ • Embeddings    │
                       │ • Similarity    │
                       │ • Search        │
                       └─────────────────┘
```

### Data Flow

1. **Document Processing**: PDFs → Text Chunks → Embeddings → Vector Store
2. **Query Processing**: User Input → Similarity Search → Context Retrieval → LLM → Response
3. **Audio Processing**: Voice Input → Whisper → Text → Query Pipeline → TTS → Audio Output

### Key Components

- **Document Loader**: PyPDFLoader for PDF text extraction
- **Text Splitter**: RecursiveCharacterTextSplitter for intelligent chunking
- **Embeddings**: OpenAI embeddings for semantic search
- **Vector Store**: FAISS for efficient similarity search
- **LLM**: OpenAI GPT-3.5-turbo for response generation
- **Audio Processing**: OpenAI Whisper (STT) and TTS models

---

## Contributing

We welcome contributions to improve the RAG Voice Assistant! Here's how you can help:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -m "Add your feature"`
5. Push to the branch: `git push origin feature/your-feature`
6. Open a Pull Request

### Areas for Contribution

- 🌍 **Internationalization**: Add support for more languages
- 🎨 **UI/UX**: Improve the frontend interface
- 🔧 **Performance**: Optimize vector search and processing
- 📱 **Mobile**: Add mobile-responsive design
- 🧪 **Testing**: Add comprehensive test coverage
- 📚 **Documentation**: Improve docs and examples

---


## Support

If you encounter any issues or have questions:

- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/rag-voice-assistant/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/rag-voice-assistant/discussions)

---

<div align="center">

**Made with ❤️ by [Hugo Parreão]**

[⭐ Star this project](https://github.com/yourusername/rag-voice-assistant) • [🍴 Fork it](https://github.com/yourusername/rag-voice-assistant/fork) • [📢 Report Issues](https://github.com/yourusername/rag-voice-assistant/issues)

</div>
