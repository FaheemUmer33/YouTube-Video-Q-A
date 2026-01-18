
# 🎥 YouTube Video Q&A with Gemini AI

An interactive Streamlit web app that allows users to **search for a YouTube video or paste a video id*, extract its **transcript**, convert it into **vector embeddings**, and ask **natural language questions** about the video. Powered by **Google Gemini**, **ChromaDB**, and **LangChain**.

---

## 🚀 Features

- 🔍 Search YouTube videos by topic or paste a direct video id.
- 📄 Automatically fetches English or Hindi transcripts using `youtube-transcript-api`.
- 🧠 Embeds the transcript using `HuggingFace` and stores in a **Chroma vector database**.
- 🤖 Uses **LangChain + Gemini 2.5 Flash** for context-aware question answering.
- 📦 Simple and responsive **Streamlit UI**.

---

## 🧱 Tech Stack

| Tool | Purpose |
|------|---------|
| `Streamlit` | UI framework |
| `YouTube Data API` | Search for videos |
| `youtube-transcript-api` | Fetch transcript |
| `LangChain` | Orchestration of RAG components |
| `Chroma` | Local vector store |
| `HuggingFace Embeddings` | Convert transcript chunks to vectors |
| `Gemini 2.5 Flash (Google Generative AI)` | Answer generation |

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/youtube-qa-gemini.git
cd youtube-qa-gemini
````

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Environment Variables

Create a `.env` file or set in your system:

```env
YOUTUBE_API_KEY=your_youtube_api_key
GOOGLE_API_KEY=your_google_api_key
```

Alternatively, export them directly:

```bash
export YOUTUBE_API_KEY=your_youtube_api_key
export GOOGLE_API_KEY=your_google_api_key
```

---

## 📦 Required Dependencies

Your `requirements.txt` might look like:

```txt
streamlit
langchain
langchain-community
langchain-google-genai
langchain-huggingface
chromadb
google-api-python-client
youtube-transcript-api
```

---

## 🖥️ How It Works

1. **User Input**:

   * Option 1: Search YouTube for a topic.
   * Option 2: Paste a full YouTube video link.

2. **Transcript Fetching**:

   * Tries English first, then Hindi.
   * Skips videos with disabled transcripts.

3. **Vectorization**:

   * Transcript is split into overlapping chunks.
   * Embeddings are generated using `all-MiniLM-L6-v2`.

4. **Vector Store**:

   * ChromaDB stores the vectors locally.
   * Enables semantic similarity search.

5. **Question Answering**:

   * Top relevant transcript chunks are passed as `context` to Gemini.
   * Gemini generates a focused answer.


## ❓ Example Questions You Can Ask

* "What is the main topic of this video?"
* "What did the speaker say about climate change?"
* "Summarize the key points from the video."

---

## 🧠 Notes

* This app uses **local vector storage**, meaning no transcript data is sent to the cloud outside of Gemini inference.
* If the video lacks a transcript or the transcript is auto-generated in an unsupported language, the app will show an error.
* Gemini only uses **provided context** and doesn’t hallucinate from outside knowledge.

---

## 🤝 Acknowledgements

* [Streamlit](https://streamlit.io/)
* [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)
* [LangChain](https://www.langchain.com/)
* [Chroma](https://www.trychroma.com/)
* [Google Gemini](https://ai.google/discover/gemini)


