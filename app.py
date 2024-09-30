import streamlit as st
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import os

# Initialize the Whisper model
model_size = "medium"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Streamlit app
st.title("Audio Translator")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

# Function to transcribe and detect language
def transcribe_audio(file):
    segments, info = model.transcribe(file, beam_size=5)
    detected_language = info.language
    segments_list = list(segments)
    transcription = " ".join([segment.text for segment in segments_list])
    return detected_language, transcription

# Function to translate text to a chosen language
def translate_text(text, target_lang):
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translator.translate(text)

# Language selection
languages = {'French': 'fr', 'Spanish': 'es', 'German': 'de', 'Italian': 'it', 'Chinese': 'zh-CN'}
output_lang = st.selectbox("Choose output language", options=list(languages.keys()))

# Process the uploaded file
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')  # Preview the audio file
    st.write("Transcribing audio...")

    # Transcribe the audio
    detected_lang, transcription = transcribe_audio(uploaded_file)

    # Display the detected language
    st.write(f"Detected Language: {detected_lang}")

    # Translate the transcription to the chosen language
    target_language_code = languages[output_lang]
    translated_text = translate_text(transcription, target_language_code)

    # Preview the translated output
    st.subheader("Translated Output")
    st.text_area("Preview", translated_text, height=300)

    # Option to download the output as a txt file
    st.download_button("Download Translation", translated_text, file_name="translated_transcription.txt")

