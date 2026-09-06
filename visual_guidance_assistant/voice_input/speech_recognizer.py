import logging

def transcribe_audio(audio_data, recognizer, language="en-US"):
    try:
        # Try Google Speech Recognition (online)
        text = recognizer.recognize_google(audio_data, language=language)
        return text
    except Exception as e:
        logging.warning(f"Google Speech Recognition failed: {e}")
        return None
