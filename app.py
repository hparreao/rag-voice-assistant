import requests
import streamlit as st
import time
import pydub
from pathlib import Path
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

PASTA_TEMP = Path(__file__).parent / 'temp'
PASTA_TEMP.mkdir(exist_ok=True)
ARQUIVO_AUDIO_TEMP = PASTA_TEMP / 'audio.mp3'
ARQUIVO_MIC_TEMP = PASTA_TEMP / 'mic.mp3'

# URL do backend
BACKEND_URL = "http://localhost:8000"

# Função para chamar o endpoint de transcrição de áudio
def transcreve_audio(caminho_audio, prompt):
    files = {'file': open(caminho_audio, 'rb')}
    data = {'prompt': prompt}  # Pode ser um prompt opcional
    response = requests.post(f"{BACKEND_URL}/audio-to-text/", files=files, data=data)
    
    if response.status_code == 200:
        return response.json().get("transcription", "")
    else:
        st.error("Erro ao transcrever áudio.")
        return ""

# Função para chamar o endpoint de texto para áudio
def texto_para_audio():
    """
    Função que converte texto inserido pelo usuário em áudio usando o backend.
    """
    texto_input = st.text_area('Digite o texto que deseja converter em áudio:', height=150)

    if st.button('Gerar Áudio'):
        if texto_input.strip():
            data = {'input_text': texto_input}
            response = requests.post(f"{BACKEND_URL}/text-to-audio/", data=data)

            if response.status_code == 200:
                with open(ARQUIVO_AUDIO_TEMP, 'wb') as f:
                    f.write(response.content)
                # Exibindo o áudio
                st.audio(str(ARQUIVO_AUDIO_TEMP), format='audio/mp3')
                st.success('Áudio gerado com sucesso!')
            else:
                st.error('Erro ao gerar áudio.')
        else:
            st.error('Por favor, insira um texto para gerar o áudio.')

if not 'transcricao_mic' in st.session_state:
    st.session_state['transcricao_mic'] = ''

@st.cache_data
def get_ice_servers():
    return [{'urls': ['stun:stun.l.google.com:19302']}]

def adiciona_chunck_de_audio(frames_de_audio, chunck_audio):
    for frame in frames_de_audio:
        sound = pydub.AudioSegment(
            data=frame.to_ndarray().tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,
            channels=len(frame.layout.channels)
        )
        chunck_audio += sound
    return chunck_audio

def transcreve_tab_mic():
    prompt_mic = st.text_input('Insira o prompt (optional)', key='input_mic')
    webrtx_ctx = webrtc_streamer(
        key='recebe_audio',
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={'video': False, 'audio': True}
    )

    if not webrtx_ctx.state.playing:
        st.write(st.session_state['transcricao_mic'])
        return

    container = st.empty()
    container.markdown('Inicie a falar...')
    chunck_audio = pydub.AudioSegment.empty()
    tempo_ultima_transcricao = time.time()
    st.session_state['transcricao_mic'] = ''
    while True:
        if webrtx_ctx.audio_receiver:
            try:
                frames_de_audio = webrtx_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                continue
            chunck_audio = adiciona_chunck_de_audio(frames_de_audio, chunck_audio)

            agora = time.time()
            if len(chunck_audio) > 0 and agora - tempo_ultima_transcricao > 10:
                tempo_ultima_transcricao = agora
                chunck_audio.export(ARQUIVO_MIC_TEMP, format='mp3')
                transcricao = transcreve_audio(ARQUIVO_MIC_TEMP, prompt_mic)
                st.session_state['transcricao_mic'] += transcricao
                container.write(st.session_state['transcricao_mic'])
                chunck_audio = pydub.AudioSegment.empty()
        else:
            break

def transcreve_tab_audio():
    prompt_input = st.text_input('Insira o prompt (optional)', key='input_audio')
    arquivo_audio = st.file_uploader('Adicione um arquivo de áudio .mp3', type=['mp3'])
    if arquivo_audio is not None:
        # Salve o arquivo temporariamente antes de enviar para o backend
        with open(ARQUIVO_AUDIO_TEMP, 'wb') as f:
            f.write(arquivo_audio.read())
        
        # Envia o arquivo de áudio para o backend para transcrição
        transcricao = transcreve_audio(ARQUIVO_AUDIO_TEMP, prompt_input)
        st.write(transcricao)

def main():
    st.header('RAG Agentic STT and TTS 🎙️', divider=True)
    st.markdown('#### IO')
    tab_mic, tab_texto, tab_audio = st.tabs(['Microfone', 'Texto para Áudio', 'Áudio para Texto'])

    with tab_mic:
        transcreve_tab_mic()
    with tab_texto:
        texto_para_audio()
    with tab_audio:
        transcreve_tab_audio()

if __name__ == '__main__':
    main()
