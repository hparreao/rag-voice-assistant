# frontend.py
import streamlit as st
import requests
import time
import queue
import pydub
from pathlib import Path
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import base64

# Constants and Setup
BACKEND_URL = "http://localhost:8000"
TEMP_DIR = Path(__file__).parent / 'temp'
TEMP_DIR.mkdir(exist_ok=True)
AUDIO_TEMP = TEMP_DIR / 'audio.mp3'
MIC_TEMP = TEMP_DIR / 'mic.mp3'

# Initialize session state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = False
    if "transcription_mic" not in st.session_state:
        st.session_state.transcription_mic = ""
    if "last_transcription_time" not in st.session_state:
        st.session_state.last_transcription_time = time.time()

# Helper Functions
def get_ice_servers():
    return [{'urls': ['stun:stun.l.google.com:19302']}]

def play_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

def process_audio_chunk(audio_file, prompt=""):
    """Process audio file for transcription"""
    files = {'file': open(audio_file, 'rb')}
    data = {'prompt': prompt}
    response = requests.post(f"{BACKEND_URL}/audio-to-text", files=files, data=data)
    if response.status_code == 200:
        return response.json().get("transcription", "")
    return ""

def add_audio_chunk(audio_frames, chunk_audio):
    """Add audio frames to chunk"""
    for frame in audio_frames:
        sound = pydub.AudioSegment(
            data=frame.to_ndarray().tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,
            channels=len(frame.layout.channels)
        )
        chunk_audio += sound
    return chunk_audio

# Core Functions
def upload_files(files):
    files_to_upload = [("files", file) for file in files]
    response = requests.post(f"{BACKEND_URL}/upload", files=files_to_upload)
    return response.status_code == 200

def query_documents(question):
    response = requests.post(
        f"{BACKEND_URL}/query",
        json={"question": question}
    )
    if response.status_code == 200:
        response_data = response.json()
        
        # Generate audio response
        audio_response = requests.post(
            f"{BACKEND_URL}/text-to-audio",
            data={"input_text": response_data["response"]}
        )
        if audio_response.status_code == 200:
            play_audio(audio_response.content)
            
        return response_data["response"]
    return None

def voice_input_tab():
    st.header("Entrada por Voz")
    prompt_mic = st.text_input('Prompt opcional para reconhecimento de voz:', key='input_mic')
    
    # Status indicator
    status_indicator = st.empty()
    transcription_output = st.empty()
    
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": get_ice_servers()},
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing:
        status_indicator.info("🎤 Gravando... Fale algo!")
        chunk_audio = pydub.AudioSegment.empty()
        
        while True:
            if webrtc_ctx.audio_receiver:
                try:
                    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                    chunk_audio = add_audio_chunk(audio_frames, chunk_audio)
                    
                    # Process every 3 seconds instead of 5
                    current_time = time.time()
                    if len(chunk_audio) >= 3000 and current_time - st.session_state.last_transcription_time >= 3:
                        st.session_state.last_transcription_time = current_time
                        
                        # Export and process audio chunk
                        chunk_audio.export(MIC_TEMP, format="mp3")
                        transcription = process_audio_chunk(MIC_TEMP, prompt_mic)
                        
                        if transcription.strip():
                            st.session_state.transcription_mic += " " + transcription
                            transcription_output.markdown(f"**Transcrição:**\n{st.session_state.transcription_mic}")
                        
                        # Reset audio chunk
                        chunk_audio = pydub.AudioSegment.empty()
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    status_indicator.error(f"Erro: {str(e)}")
                    break
            else:
                break
    else:
        if st.session_state.transcription_mic:
            transcription_output.markdown(f"**Última transcrição:**\n{st.session_state.transcription_mic}")
            
        # Add clear transcription button
        if st.button("Limpar Transcrição"):
            st.session_state.transcription_mic = ""
            transcription_output.empty()

# Tab Functions
def chat_tab():
    st.header("Chat Interface")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Digite sua pergunta:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if not st.session_state.uploaded_files:
            st.warning("Por favor, faça o upload de um PDF primeiro.", icon="🚨")
            return
        
        with st.spinner("Processando..."):
            response = query_documents(prompt)
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("Erro ao processar sua pergunta")

def text_to_speech_tab():
    st.header("Texto para Áudio")
    text_input = st.text_area("Digite o texto para converter em áudio:", height=150)
    voice = st.selectbox(
        "Selecione a voz:",
        ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    )
    
    if st.button("Gerar Áudio"):
        if text_input.strip():
            with st.spinner("Gerando áudio..."):
                response = requests.post(
                    f"{BACKEND_URL}/text-to-audio",
                    data={"input_text": text_input, "voice": voice}
                )
                if response.status_code == 200:
                    st.audio(response.content, format="audio/mp3")
                    st.success('Áudio gerado com sucesso!')
                else:
                    st.error('Erro ao gerar áudio.')
        else:
            st.error('Por favor, digite algum texto para gerar o áudio.')

def audio_to_text_tab():
    st.header("Áudio para Texto")
    prompt_input = st.text_input('Prompt opcional para transcrição:', key='input_audio')
    uploaded_audio = st.file_uploader('Upload de arquivo de áudio (.mp3)', type=['mp3'])
    
    if uploaded_audio:
        with open(AUDIO_TEMP, 'wb') as f:
            f.write(uploaded_audio.read())
        
        transcription = process_audio_chunk(AUDIO_TEMP, prompt_input)
        st.write(transcription)

def main():
    st.title("Chat RAG com Interação por Voz 🎙️")
    initialize_session_state()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("Gerenciamento de Documentos")
        with st.form("upload-form", clear_on_submit=True):
            uploaded_files = st.file_uploader(
                "Faça o Upload do seu PDF:",
                accept_multiple_files=True,
                type=["pdf"]
            )
            submit_button = st.form_submit_button("Processar")
            
            if submit_button and uploaded_files:
                with st.spinner("Processando documentos..."):
                    if upload_files(uploaded_files):
                        st.session_state.uploaded_files = True
                        st.success("Upload realizado com sucesso!", icon="✅")
                    else:
                        st.error("Erro ao processar documentos")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Chat",
        "Entrada por Voz",
        "Texto para Áudio",
        "Áudio para Texto"
    ])
    
    with tab1:
        chat_tab()
    with tab2:
        voice_input_tab()
    with tab3:
        text_to_speech_tab()
    with tab4:
        audio_to_text_tab()

if __name__ == "__main__":
    main()
