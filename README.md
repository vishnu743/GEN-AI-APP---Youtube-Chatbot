# GEN-AI-APP---Youtube-Chatbot


##  Overview

This project is a **Generative AI-powered YouTube Chatbot** that allows users to interact with YouTube video content using natural language.

Users can ask questions about a video, and the chatbot provides meaningful answers based on the video's content.

---

##  Features

*  Ask questions about any YouTube video
*  AI-powered responses using LLM
*  Extracts and processes video transcripts
*  Fast and interactive chat interface
*  Context-aware answers

---

##  Tech Stack

* **Python**
* **OpenAI / LLM APIs**
* **YouTube Transcript API**
* **Streamlit / Flask (Frontend)**
* **LangChain (optional)**

---

##  How It Works

1. User provides a YouTube video link
2. The app extracts the video transcript
3. Text is processed and converted into embeddings
4. User asks questions
5. AI generates answers based on the video content

---

##  Project Structure

```
GEN-AI-APP---Youtube-Chatbot/
│── app.py
│── requirements.txt
│── utils/
│   └── transcript.py
│── chatbot/
│   └── model.py
│── README.md
```

---

##  Installation

```bash
git clone https://github.com/YOUR_USERNAME/GEN-AI-APP---Youtube-Chatbot.git
cd GEN-AI-APP---Youtube-Chatbot
pip install -r requirements.txt
```

---

##  Run the App

```bash
streamlit run app.py
```

---

##  Demo

(Add screenshots or demo GIF here)


