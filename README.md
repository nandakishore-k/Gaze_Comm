# OptiType: Eye-Controlled Virtual Keyboard with Word Prediction and Text-to-Speech

OptiType is an assistive communication system designed for users with limited mobility. The application allows hands-free typing through **gaze and blink detection**, provides **real-time word prediction** using **NLTK & Trie-based models**, and converts selected text to speech using TTS systems.

## ✨ Features

- 👁️ **Gaze and Blink-Based Navigation** using OpenCV and Dlib
- 🧠 **Next Word Prediction** with NLP models (DistilGPT2 / NLTK / Trie)
- 📖 **Partial Word Auto-completion** using Trie data structure
- 🔊 **Text-to-Speech Output** for spoken communication
- 💻 **Intuitive GUI** built using PyQt5
- 📦 **Packaged as a Desktop App** using PyInstaller

## 🛠️ Tech Stack

- **Python 3**
- **OpenCV, Dlib** – for facial landmark detection
- **PyQt5** – GUI interface
- **Transformers (Hugging Face), NLTK** – NLP word prediction
- **Pyttsx3 / gTTS** – Text-to-Speech
- **Trie Data Structure** – Custom auto-completion engine

## 📂 Project Structure

```bash
├── optitype/
│   ├── main.py                  # Entry point for the application
│   ├── gui/                     # PyQt5 UI layout and control files
│   ├── nlp/                     # Word prediction modules (Trie, NLTK, Transformers)
│   ├── assets/                  # shape_predictor.dat, nltk_data, and dictionary files
│   └── utils/                   # Helper functions and shared utilities
