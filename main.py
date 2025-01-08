from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from openai import OpenAI
from pathlib import Path
import shutil
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configuração
PASTA_TEMP = Path("./temp")
PASTA_TEMP.mkdir(exist_ok=True)
VECTOR_DB_DIR = Path("./vectorstore")
VECTOR_DB_DIR.mkdir(exist_ok=True)

app = FastAPI()
client = OpenAI()

# Inicialização do ChromaDB
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory=str(VECTOR_DB_DIR), embedding_function=embedding_model)

# Inicialização do cliente OpenAI para geração de texto
persistent_client = chromadb.PersistentClient()

# Função para gerar uma resposta com base nos documentos recuperados
def gerar_resposta_com_documentos(query, documentos):
    # concatenando os documentos para dar contexto ao modelo
    contexto = "\n".join([doc['text'] for doc in documentos['documents']])
    
    # consulta e o contexto para o modelo de linguagem para gerar uma resposta
    prompt = f"Com base no seguinte contexto, responda à consulta:\n\n{contexto}\n\nConsulta: {query}"
    
    response = client.Completion.create(
        model="text-davinci-003", 
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# Endpoints
@app.post("/text-to-audio/")
async def text_to_audio(input_text: str = Form(...)):
    """Recebe um texto e retorna um arquivo de áudio gerado."""
    if not input_text.strip():
        raise HTTPException(status_code=400, detail="Texto não pode ser vazio.")
    
    response = client.audio.speech.create(model="tts-1", voice="alloy", input=input_text)
    output_path = PASTA_TEMP / "output_audio.mp3"
    response.stream_to_file(output_path)
    return FileResponse(output_path, media_type="audio/mp3")

@app.post("/audio-to-text/")
async def audio_to_text(file: UploadFile):
    """Recebe um áudio e retorna a transcrição em texto."""
    if file.content_type != "audio/mpeg":
        raise HTTPException(status_code=400, detail="O arquivo deve estar no formato MP3.")
    
    temp_file = PASTA_TEMP / file.filename
    with temp_file.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    with temp_file.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            language="pt",
            response_format="text",
            file=audio_file
        )
    return JSONResponse(content={"transcription": transcription})

@app.post("/upsert-documents/")
async def upsert_documents(file: UploadFile):
