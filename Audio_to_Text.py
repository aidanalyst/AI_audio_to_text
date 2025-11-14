import streamlit as st
import whisper
import tempfile
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")


# --------------------------------------------------
# 🧭 PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="🎧 AI Transcription App",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Multi-Language Audio Transcription App")
st.markdown("Upload an audio file and transcribe it using OpenAI Whisper with translation support.")

# --------------------------------------------------
# ⚙️ SIDEBAR SETTINGS
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # Whisper model size
    model_size = st.selectbox(
        "Select Whisper Model",
        options=["tiny", "base", "small", "medium", "large"],
        index=2,
        help="Larger models give better accuracy but are slower. 'base' is a good balance."
    )

    # Supported languages
    languages = {
        "Auto-detect": None,
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Dutch": "nl",
        "Russian": "ru",
        "Chinese": "zh",
        "Japanese": "ja",
        "Korean": "ko",
        "Arabic": "ar",
        "Hindi": "hi",
        "Turkish": "tr",
        "Polish": "pl",
        "Swedish": "sv",
        "Danish": "da",
        "Norwegian": "no",
        "Finnish": "fi"
    }

    selected_language = st.selectbox(
        "Audio Language",
        options=list(languages.keys()),
        help="Choose the language or use auto-detect"
    )

    translate_to_english = st.checkbox(
        "Translate to English",
        help="Translates non-English speech into English text"
    )

    st.markdown("---")
    st.info("🎵 **Supported Audio Formats:** MP3, WAV, M4A, FLAC, OGG, MP4")

# --------------------------------------------------
# 🧠 MODEL CACHING FUNCTION
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str):
    """Load and cache Whisper model to avoid repeated downloads."""
    model = whisper.load_model(model_size)
    return model

# --------------------------------------------------
# 📁 FILE UPLOAD SECTION
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload an audio file to transcribe",
    type=["mp3", "wav", "m4a", "flac", "ogg", "webm", "mp4"]
)

# --------------------------------------------------
# 🚀 MAIN LOGIC
# --------------------------------------------------
if uploaded_file:
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

    if st.button("🎯 Transcribe Audio", type="primary"):
        try:
            with st.spinner("Loading Whisper model..."):
                model = load_whisper_model(model_size)

            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Perform transcription
            with st.spinner("Transcribing... this may take a few minutes ⏳"):
                language_code = languages[selected_language]
                task_type = "translate" if translate_to_english else "transcribe"

                result = model.transcribe(
                    tmp_path,
                    language=language_code,
                    task=task_type,
                    verbose=False
                )

            os.unlink(tmp_path)  # Clean up

            # --------------------------------------------------
            # ✅ DISPLAY RESULTS
            # --------------------------------------------------
            st.success("✅ Transcription completed successfully!")

            detected_lang = result.get("language", "Unknown").upper()
            st.info(f"**Detected Language:** {detected_lang}")

            # Full transcription text
            st.subheader("📝 Transcription Output")
            st.text_area(
                "Transcribed Text",
                value=result["text"],
                height=300,
                label_visibility="collapsed"
            )

            # --------------------------------------------------
            # 💾 DOWNLOAD OPTIONS
            # --------------------------------------------------
            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="📥 Download as TXT",
                    data=result["text"],
                    file_name=f"{uploaded_file.name}_transcription.txt",
                    mime="text/plain"
                )

            # Optional: SRT format
            with col2:
                srt_output = ""
                for i, seg in enumerate(result["segments"], 1):
                    start = seg["start"]
                    end = seg["end"]
                    text = seg["text"].strip()
                    srt_output += f"{i}\n{int(start//60):02}:{int(start%60):02}:00 --> {int(end//60):02}:{int(end%60):02}:00\n{text}\n\n"

                st.download_button(
                    label="🎬 Download as SRT",
                    data=srt_output,
                    file_name=f"{uploaded_file.name}_subtitles.srt",
                    mime="text/plain"
                )

            # --------------------------------------------------
            # 🕒 SEGMENT DETAILS
            # --------------------------------------------------
            with st.expander("🔍 View Segments with Timestamps"):
                for seg in result["segments"]:
                    start_time = f"{int(seg['start']//60)}:{int(seg['start']%60):02d}"
                    end_time = f"{int(seg['end']//60)}:{int(seg['end']%60):02d}"
                    st.markdown(f"**[{start_time} - {end_time}]** {seg['text']}")

        except Exception as e:
            st.error(f"❌ An error occurred during processing: {e}")
            st.info("Try reinstalling dependencies: `pip install openai-whisper streamlit`")

# --------------------------------------------------
# 📖 FOOTER GUIDE
# --------------------------------------------------
st.markdown("---")
st.markdown("""
### 🧭 How to Use
1. Choose your Whisper model (larger = more accurate but slower)
2. Select or auto-detect the audio language
3. Upload an audio file
4. Click **Transcribe Audio**
5. Download results as TXT or SRT

💡 **Note:** The first model load may take longer due to model download.
""")
