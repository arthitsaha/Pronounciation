import streamlit as st
import speech_recognition as sr
from faster_whisper import WhisperModel
from difflib import SequenceMatcher
import tempfile
import os

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Initialize the Whisper model
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_whisper_model()

def record_audio():
    with sr.Microphone() as source:
        st.write("Speak now...")
        audio = recognizer.listen(source)
    return audio

def transcribe_audio(audio_file):
    try:
        segments, _ = model.transcribe(audio_file)
        transcription = " ".join([segment.text for segment in segments])
        return transcription.lower()
    except Exception as e:
        st.error(f"Error in transcription: {e}")
        return ""

def evaluate_pronunciation(spoken_text, correct_text):
    spoken_text = spoken_text.lower()
    correct_text = correct_text.lower()
    
    similarity = SequenceMatcher(None, spoken_text, correct_text).ratio()
    
    spoken_words = spoken_text.split()
    correct_words = correct_text.split()
    
    feedback = []
    for spoken, correct in zip(spoken_words, correct_words):
        if spoken != correct:
            feedback.append(f"'{spoken}' should be '{correct}'")
    
    return similarity, feedback

def provide_feedback(similarity, feedback):
    if similarity >= 0.9:
        st.success("Excellent pronunciation!")
    elif similarity >= 0.7:
        st.info("Good pronunciation, but there's room for improvement.")
    else:
        st.warning("Your pronunciation needs work. Let's practice more!")
    
    if feedback:
        st.write("Specific feedback:")
        for item in feedback:
            st.write(f"- {item}")

def main():
    st.title("Pronunciation Evaluation App")

    correct_text = st.text_input("Enter the text to practice:")
    
    if st.button("Start Recording"):
        with st.spinner("Recording..."):
            audio = record_audio()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio.get_wav_data())
            temp_audio_path = temp_audio.name

        with st.spinner("Transcribing..."):
            spoken_text = transcribe_audio(temp_audio_path)
        
        st.write(f"You said: '{spoken_text}'")
        
        similarity, feedback = evaluate_pronunciation(spoken_text, correct_text)
        provide_feedback(similarity, feedback)

        # Clean up the temporary file
        os.unlink(temp_audio_path)

if __name__ == "__main__":
    main()