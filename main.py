import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os


# --- CONFIG ---

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY")


# --- SETUP LLM ---
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

# --- PROMPT TEMPLATE ---
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=['context', 'question']
)

# --- YouTube Search ---
def search_youtube(query, max_results=1):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(
        q=query,
        part="id,snippet",
        maxResults=max_results,
        type="video"
    )
    response = request.execute()
    if response['items']:
        item = response['items'][0]
        return (
            item['id']['videoId'],
            item['snippet']['title'],
            item['snippet']['channelTitle']
        )
    return None, None, None

# --- Load Transcript ---
def load_transcript(video_id):
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

    try:
        # Try fetching English transcript first
        return "\n".join([entry.text for entry in YouTubeTranscriptApi().fetch(video_id, languages=['en'])])
    except NoTranscriptFound:
        try:
            return "\n".join([entry.text for entry in YouTubeTranscriptApi().fetch(video_id, languages=['hi'])])
            # Get list of available transcripts      
        except Exception as e:
            raise Exception(f"Transcript error: {e}")
    except TranscriptsDisabled:
        raise Exception("Transcripts are disabled for this video.")

# --- Create VectorStore ---
def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embedding=embeddings)

# --- STREAMLIT UI ---
st.title("🎥 YouTube Video Q&A")

option = st.radio("Choose input method:", ["Paste Video ID", "Search Topic"])

# --- Choose Input ---
if option == "Paste Video ID":
    st.session_state.video_id= st.text_input("Paste YouTube Video link")
  
else:
    topic = st.text_input("Enter a topic to search")
    if st.button("Search Video"):
        video_id, title, channel = search_youtube(topic)
        if video_id:
            st.session_state.video_id = video_id
            st.session_state.video_title = title
            st.session_state.video_channel = channel
            st.success(f"Top result: {title} by {channel}")
        else:
            st.error("No video found for that topic.")

# --- Show selected video info ---
if option == "Search Topic" and "video_title" in st.session_state:
    st.info(f"Selected: {st.session_state.video_title} by {st.session_state.video_channel}")

# --- Process Transcript and Vector DB ---
if st.session_state.get("video_id", ""):
    if st.button("Process Video"):
        try:
            st.info("Fetching transcript...")
            full_text = load_transcript(st.session_state.video_id)
            st.success("Transcript fetched.")

            st.info("Creating vectorstore...")
            vectorstore = create_vectorstore(full_text)
            st.session_state.retriever = vectorstore.as_retriever(search_type="similarity")
            st.success("Vectorstore ready! You can now ask questions about the video.")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Q&A Interface ---
if 'retriever' in st.session_state:
    user_q = st.text_input("Ask a question about the video")
    if user_q:
        with st.spinner("Thinking..."):
            retrieved_docs = st.session_state.retriever.invoke(user_q)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            final_prompt = prompt.format(context=context_text, question=user_q)
            answer = llm.invoke(final_prompt)
            st.success(answer.content)

st.markdown("---")
st.caption("Built with Gemini")
