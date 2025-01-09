from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import shutil
from typing import List, Optional
import uvicorn
import logging
from pathlib import Path
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Verify OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize global variables
UPLOAD_DIR = Path("uploaded_pdfs")
TEMP_DIR = Path("temp_audio")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Store vectorstore in memory
vectorstore = None

class Query(BaseModel):
    question: str

# [Previous functions remain the same: format_docs, load_prompt, save_upload_file, process_pdfs]
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_prompt():
    prompt = """É necessário responder à pergunta na frase, tal como no conteúdo do pdf.
    O contexto e a pergunta do utilizador são apresentados a seguir.
    Contexto = {context}
    Pergunta = {question}
    Se a resposta não estiver no pdf, responda "Não consigo responder a essa pergunta com minha base de informações"
    """
    return ChatPromptTemplate.from_template(prompt)

async def save_upload_file(upload_file: UploadFile) -> Path:
    """Save an uploaded file and return its path."""
    try:
        file_path = UPLOAD_DIR / upload_file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return file_path
    except Exception as e:
        logger.error(f"Error saving file {upload_file.filename}: {str(e)}")
        raise

def process_pdfs(file_paths: List[Path]):
    """Process PDF files and create vectorstore."""
    try:
        text_chunks = []
        for file_path in file_paths:
            logger.info(f"Processing PDF: {file_path}")
            loader = PyPDFLoader(str(file_path))
            chunks = loader.load_and_split(
                text_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=512,
                    chunk_overlap=30,
                    length_function=len,
                    separators=["\n\n", "\n", ".", " "]
                )
            )
            text_chunks.extend(chunks)
            logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
        
        if not text_chunks:
            raise ValueError("No text chunks extracted from PDFs")
        
        embeddings = OpenAIEmbeddings()
        return FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    except Exception as e:
        logger.error(f"Error processing PDFs: {str(e)}")
        raise

# [Previous endpoints remain the same: upload_files, query_documents]
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    global vectorstore
    saved_files = []
    
    try:
        # Validate files
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save files
        for file in files:
            file_path = await save_upload_file(file)
            saved_files.append(file_path)
            logger.info(f"Saved file: {file_path}")
        
        # Process PDFs
        vectorstore = process_pdfs(saved_files)
        logger.info("Vectorstore created successfully")
        
        # Cleanup
        for file_path in saved_files:
            file_path.unlink()
            logger.info(f"Cleaned up file: {file_path}")
        
        return {"message": "Files processed successfully"}
    
    except Exception as e:
        logger.error(f"Error in upload_files: {str(e)}")
        # Cleanup on error
        for file_path in saved_files:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Cleaned up file on error: {file_path}")
        
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(query: Query):
    if not vectorstore:
        raise HTTPException(status_code=400, detail="No documents have been uploaded yet")
    
    try:
        logger.info(f"Processing query: {query.question}")
        
        # Create similarity search
        similar_embeddings = vectorstore.similarity_search(query.question)
        similar_store = FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings())
        
        # Set up RAG chain
        retriever = similar_store.as_retriever()
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        prompt = load_prompt()
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Get response
        response = rag_chain.invoke(query.question)
        logger.info("Query processed successfully")
        return {"response": response}
    
    except Exception as e:
        logger.error(f"Error in query_documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# New audio endpoints
@app.post("/text-to-audio")
async def text_to_audio(input_text: str = Form(...), voice: Optional[str] = Form("alloy")):
    """Convert text to audio using OpenAI's TTS API."""
    try:
        logger.info(f"Converting text to audio with voice: {voice}")
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=input_text,
            speed=1.0
        )
        
        output_path = TEMP_DIR / "output_audio.mp3"
        response.stream_to_file(str(output_path))
        
        logger.info("Audio file created successfully")
        return FileResponse(str(output_path), media_type="audio/mp3")
    except Exception as e:
        logger.error(f"Error in text_to_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio-to-text")
async def audio_to_text(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form("")
):
    """Convert audio to text using OpenAI's Whisper API."""
    try:
        if not file.filename.endswith('.mp3'):
            raise HTTPException(status_code=400, detail="File must be MP3 format")
        
        logger.info(f"Processing audio file: {file.filename}")
        temp_path = TEMP_DIR / "temp_audio.mp3"
        
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        transcription_options = {
            "model": "whisper-1",
            "file": open(temp_path, "rb"),
            "language": "pt",
            "response_format": "text"
        }
        
        if prompt:
            transcription_options["prompt"] = prompt
            logger.info(f"Using prompt: {prompt}")
        
        transcription = client.audio.transcriptions.create(**transcription_options)
        
        # Clean up temporary file
        temp_path.unlink()
        logger.info("Audio transcription completed successfully")
        
        return {"transcription": transcription}
    except Exception as e:
        logger.error(f"Error in audio_to_text: {str(e)}")
        # Ensure cleanup on error
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)