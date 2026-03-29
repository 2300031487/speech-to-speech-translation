# Speech-to-Speech Translation System

## About the Project

This project is a simple speech-to-speech translation system. The idea is to take spoken input, convert it into text, translate it into another language, and then generate speech output in that language.

It combines speech recognition, translation, and text-to-speech into one pipeline so the user can speak and hear the translated result.


## What it does

* Takes voice input from the user
* Converts speech to text using Whisper
* Detects the language automatically
* Translates the text into a selected language
* Converts translated text back into speech



## How it works

The flow of the system is:

Audio → Noise Reduction → Speech Recognition → Translation → Text-to-Speech → Output



## Technologies used

* Faster-Whisper (for speech recognition)
* Deep Translator (for translation)
* Edge TTS (for natural voice output)
* Gradio (for UI)
* PyTorch (for model execution)
* SoundFile, NumPy (for audio processing)



## How to run

1. Install dependencies:


pip install -r requirements.txt

2. Run the app:

python app.py

3. Open the link shown in terminal and start using the app.


## Notes

* Speak clearly for better accuracy
* Short sentences (5–7 words) give better results
* Works faster if GPU is available


## Future improvements

* Real-time streaming instead of short audio input
* Support for more languages
* Better translation models
* Lower latency


## Use cases

* Talking to people who speak different languages
* Travel communication
* Accessibility tools


## Author

Bandaru Charan
