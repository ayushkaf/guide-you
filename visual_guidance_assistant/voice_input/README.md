# Voice Input Module

This package provides microphone input and speech-to-text transcription for the Visual Guidance Assistant.

- `microphone.py`: Threaded microphone listener, pushes recognized speech to a command queue.
- `speech_recognizer.py`: Handles audio-to-text transcription using SpeechRecognition.

## Usage

Instantiate `MicrophoneListener` with a command queue and config. Start the thread to enable real-time voice input.