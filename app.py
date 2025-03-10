"""
AI-Powered YouTube Video Summarizer

Description:
An advanced AI pipeline designed to summarize YouTube videos instantly. It integrates yt_dlp, OpenAI Whisper, and GPT-4o-Mini into a user-friendly Streamlit interface.

Technologies:
- Python
- Streamlit
- yt_dlp
- OpenAI Whisper
- GPT-4o-Mini
"""




import streamlit as st
import whisper
import os
import rich
from rich.progress import Progress
import ffmpeg
import openai
import os
import json
import sys
import yt_dlp
import configparser
import warnings
import streamlit as st
import time



warnings.filterwarnings("ignore")
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
model = whisper.load_model("small")



def load_openai_key(config_path="app.properties"):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config["DEFAULT"]["openai_api_key"]

def download_youtube_audio(youtube_url, output_folder, file_name):

    """
        Downloads audio from a YouTube video and converts it to MP3.

        Parameters:
        - youtube_url: URL of the YouTube video.
        - output_folder: Directory to store extracted audio and metadata.
        - file_name: Base name for audio and metadata files.

        Returns:
            mp3_filepath (str): Path to downloaded MP3 audio.
    """

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # yt-dlp options to extract only audio and convert it to mp3
    ydl_opts = {
        'outtmpl': os.path.join(output_folder, file_name+'.%(ext)s'),
        'format': 'bestaudio/best',   # highest quality audio
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # This downloads the video and returns metadata
        info_dict = ydl.extract_info(youtube_url, download=True)
    
    # The final MP3 filename (after conversion by FFmpegExtractAudio)
    # will be in the same folder with the ".mp3" extension.
    # `title` is auto-extracted from the info_dict.
    title = info_dict.get('title', 'no_title')
    mp3_filename = os.path.join(output_folder, f"{file_name}.mp3")
    print("######### mp3_filename = ", mp3_filename)
    
    # Save metadata to JSON
    metadata_filename = os.path.join(output_folder, f"{file_name}_metadata.json")
    with open(metadata_filename, 'w', encoding='utf-8') as f:
        json.dump(info_dict, f, indent=4, ensure_ascii=False)

    print(f"MP3 saved at: {mp3_filename}")
    print(f"Metadata saved at: {metadata_filename}")
    return mp3_filename, metadata_filename



def transribe_audio(model, audio_file_path):

    """
    Converts audio file into text transcript using Whisper.

    Args:
        model: Whisper ASR model instance.
        audio_filepath (str): Path to MP3 file.

    Returns:
        transcript_text (str): Textual transcript of the audio.
    """

    result = model.transcribe(audio_file_path)
    transcribe_file_path = os.path.splitext(audio_file_path)[0] + "_transcribe.json"
    with open(transcribe_file_path, "w", encoding="utf-8") as outfile:
        json.dump(result, outfile, indent=4)
        print("Saved the transribe file in the below path = ", outfile)
    return result['text'], outfile



def summarize_text(text, output_folder, file_name, model="gpt-4o-mini"):


    """
    Summarizes given text using GPT-4o-Mini.

    Args:
        text (str): Text to summarize.
        model (str): OpenAI model to use for summarization.

    Returns:
        summary (str): Generated summary text.
    """

    openai_api_key = load_openai_key("app.properties")
    client = openai.OpenAI(api_key=openai_api_key) 
    prompt = f"""
    You are an expert in technical writing. Summarize the following text in a structured manner with headings, subheadings, and bullet points. If there are any implicit diagrams or visual structures, convert them into text-based representations.
    
    Text to summarize:
    {text}
    
    Your summary should be well-organized, clear, and concise.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant skilled in summarization."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    summary = response.choices[0].message.content

    summarize_file_path = output_folder+"/"+file_name + "_summary.text"
    with open(summarize_file_path, "w", encoding="utf-8") as outfile:
        outfile.write(summary)
        print("Saved the summarized file in the below path = ", outfile)
        
    return summary, outfile




def process_video(youtube_url, project_name):
    """
    Orchestrates the video summarization pipeline.

    Parameters:
        youtube_url (str): URL of the YouTube video.
        project_name (str): Name of the current summarization project.

    Returns:
        summary (str): Generated summary of video content.
    """
    
    video_name = project_name
    output_folder = os.getcwd()+"/"+video_name

    # Initialize progress bar at 0%
    progress_bar = st.progress(0)

    # --- Step 1: Download Video ---
    step1_placeholder = st.empty()
    with step1_placeholder:
        with st.spinner("Downloading Video..."):
            mp3_filename, metadata_filename = download_youtube_audio(youtube_url=youtube_url, output_folder = output_folder, file_name = video_name)
        st.success("Video downloaded successfully!")
        time.sleep(0.8)
    step1_placeholder.empty()
    progress_bar.progress(33)

    # --- Step 2: Transcribe Video ---
    step2_placeholder = st.empty()
    with step2_placeholder:
        with st.spinner("Transcribing Video..."):
            transcirbe_result = transribe_audio(model, mp3_filename)
        st.success("Transcription complete!")
        time.sleep(0.8)
    step2_placeholder.empty()
    progress_bar.progress(66)

    # --- Step 3: Summarize Text ---
    step3_placeholder = st.empty()
    with step3_placeholder:
        with st.spinner("Summarizing Text..."):
            summary, summary_file_path = summarize_text(transcirbe_result, output_folder = output_folder, file_name=video_name)
        st.success("Summary complete!")
        time.sleep(0.8)
    step3_placeholder.empty()
    progress_bar.progress(100)

    # Remove the progress bar after completion
    time.sleep(0.5)
    progress_bar.empty()

    return summary


# Set page configuration
st.set_page_config(page_title="YouTube Summarizer", layout="centered")

# Define a fixed color that will be used for both the button and the progress bar
primary_color = "white"  # Base color for heading and progress bar

# Inject custom CSS for enhanced styling
st.markdown(f"""
    <style>
    /* Overall page background and font */
    .stApp {{
        background: white;
        font-family: "Helvetica Neue", Arial, sans-serif;
    }}

    /* Main container styling (card-like container) */
    .main > div {{
        display: flex;
        flex-direction: column;
        align-items: center;
        max-width: 650px;
        width: 100%;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.05);
    }}

    /* Heading style with matching background color */
    .app-heading {{
        background-color: {primary_color};
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        text-align: center;
        width: 100%;
        margin-bottom: 1rem;
    }}

    /* Subtitle text styling */
    .stMarkdown p {{
        text-align: center;
        color: #455a64;
    }}

    /* Style text input boxes */
    .stTextInput > label {{
        font-weight: 600;
        font-size: 0.95rem;
        color: #374151;
    }}
    input[type="text"] {{
        border: 1px solid #cbd5e1;
        border-radius: 8px;
    }}

    /* Beautiful "Process Video" button styling */
    div.stButton > button {{
        background: #1D4ED8;
        color: white;
        padding: 0.6em 1.5em;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.3s ease;
    }}
    div.stButton > button:hover {{
        background: #1D4ED8;
    }}

    /* Summary box styling */
    .summary-box {{
        width: 100%;
        background-color: #F1F0E9;
        padding: 1em;
        border-radius: 8px;
        margin-top: 1em;
        line-height: 1.5;
    }}

    /* Custom progress bar color matching the primary color */
    [data-testid="stProgressbar"] > div > div {{
        background-color: {primary_color};
    }}
    </style>
    """, unsafe_allow_html=True)



def main():
    # Display heading with a matching background color
    st.markdown('<h1 class="app-heading">YouTube Video Summarizer</h1>', unsafe_allow_html=True)
    st.write("""
    Enter a YouTube link and a project name to process and summarize your video.
    Follow the progress via the progress bar below.
    """)

    # Input fields for YouTube URL and Project Name
    youtube_url = st.text_input("Enter YouTube Video Link")
    project_name = st.text_input("Enter Project Name")

    # "Process Video" button
    if st.button("Process Video"):
        if youtube_url and project_name:
            summary = process_video(youtube_url, project_name)
            # Display final summary in a styled box
            st.markdown(
                f"""
                <div class="summary-box">
                    <h4>Summary for '{project_name}'</h4>
                    <p>{summary}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("Please provide both a valid YouTube link and a Project Name.")

if __name__ == "__main__":
    main()
