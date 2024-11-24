import requests
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from dotenv import load_dotenv
import openai
import difflib
import re
from supabase import create_client
from supabase.lib.client_options import ClientOptions
import logging  # Import the logging module
import json
from openai import AzureOpenAI
from typing import List, Dict
from pytubefix import YouTube
from pytubefix.cli import on_progress
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from openai import OpenAI

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"


# Use GPT-4 for summarization
model = ChatOpenAI(model="gpt-4o")

# # Vimeo API credentials
# access_token = os.getenv('ACCESS_TOKEN')
# client_id = os.getenv('CLIENT_ID')
# client_secret = os.getenv('CLIENT_SECRET')

# Vimeo API base URL
api_base_url = "https://api.vimeo.com"

# Global variables
video_id_global = None
node_id_global = None
video_filename = None
audio_filename = None
compressed_audio_filename = None
transcript = None
segments = None
summary = None

def make_supabase_client(schema="public"):
    opts = ClientOptions().replace(schema=schema)
    return create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY"),
        options=opts
    )

def update_failedDownload_status(video_id, table_name="videos", schema="public"):
    supabase_client = make_supabase_client(schema=schema)
    update_values = {"status": "FAILEDDOWNLOAD"}
    filter_conditions = {"video_id": video_id}
    supabase_client.table(table_name).update(update_values).match(filter_conditions).execute()

    # Attempt to remove files with error handling
    for filename in [video_filename, audio_filename, compressed_audio_filename]:
        try:
            os.remove(filename)
        except PermissionError:
            logging.error(f"PermissionError: Cannot delete {filename}. It may be in use by another process.")
        except FileNotFoundError:
            logging.warning(f"FileNotFoundError: {filename} does not exist.")

def update_failedURL_status(video_id, table_name="videos", schema="public"):
    supabase_client = make_supabase_client(schema=schema)
    update_values = {"status": "FAILEDURL"}
    filter_conditions = {"video_id": video_id}
    supabase_client.table(table_name).update(update_values).match(filter_conditions).execute()
    
def update_failed_status(video_id, table_name="videos", schema="public"):
    supabase_client = make_supabase_client(schema=schema)
    update_values = {"status": "FAILED"}
    filter_conditions = {"video_id": video_id}
    supabase_client.table(table_name).update(update_values).match(filter_conditions).execute()
    
    # Attempt to remove files with error handling
    for filename in [video_filename, audio_filename, compressed_audio_filename]:
        try:
            os.remove(filename)
        except PermissionError:
            logging.error(f"PermissionError: Cannot delete {filename}. It may be in use by another process.")
        except FileNotFoundError:
            logging.warning(f"FileNotFoundError: {filename} does not exist.")

async def download_vimeo_video_step(video_id: str):
    global video_filename
    global video_id_global
    global node_id_global

    video_id_global = video_id
    logging.info(f"Downloading video with ID: {video_id}")

     # Fetch the URL from the database using video_id
    supabase_client = make_supabase_client()
    response = supabase_client.table("videos").select("url").eq("video_id", video_id).execute()

    
    if response.data and len(response.data) > 0:
        video_url = response.data[0]['url']
        logging.info(f"Fetched video URL: {video_url}")

        try:
            # Check if the URL is from Vimeo
            if "vimeo.com" in video_url:

                return None
                
            elif "youtube.com" in video_url or "youtu.be" in video_url:
                # YouTube download logic
                # YouTube download logic using pytubefix
                youtube_url = video_url
                yt = YouTube(youtube_url, on_progress_callback=on_progress)

                logging.info(f"Video Title: {yt.title}")

                # Get the highest resolution stream available
                ys = yt.streams.get_highest_resolution()

                # Download the video
                logging.info(f"Downloading video: {yt.title} at highest resolution")
                ys.download()

                video_filename = f"{yt.title}.mp4"
                logging.info(f"Downloaded video saved as: {video_filename}")

                return video_filename
                # ... existing YouTube download logic ...
            else:
                # Handle case for unsupported URLs
                logging.error(f"Unsupported URL: {video_url}")
                raise ValueError("Unsupported URL encountered, stopping the chain.")  # Raise an exception

        except Exception as e:
            logging.error(f"Error downloading video: {e}")
            update_failed_status(video_id_global)
            raise e
    else:
        logging.error(f"No URL found for video ID: {video_id}")
        update_failed_status(video_id_global)  # Update status to URL Failed


def sanitize_filename(filename: str) -> str:
    """
    Ensures that the filename is compatible with Windows by removing or replacing
    invalid characters like '|'.
    """
    # Replace invalid characters like '|' with a space or another safe character
    return filename.replace('|', ' ').replace('  ', ' ')

def extract_audio_step(video_filename: str):
    global audio_filename
    logging.info(f"Extracting audio from video: {video_filename}")
    
    try:
        # Sanitize the filename to avoid issues with special characters
        sanitized_video_filename = sanitize_filename(video_filename)
        
        # Create the audio filename by changing the extension to .wav
        audio_filename = ".".join(sanitized_video_filename.split(".")[:-1]) + ".wav"
        
        # Extract the audio from the video using the sanitized filename
        video_clip = VideoFileClip(sanitized_video_filename)
        audio_clip = video_clip.audio
        
        # Write the audio as a .wav file
        audio_clip.write_audiofile(audio_filename)
        
        logging.info(f"Audio extracted and saved as: {audio_filename}")
        
        return audio_filename
    
    except Exception as e:
        logging.error(f"Error extracting audio: {e}")
        update_failed_status(video_id_global)  # Ensure video_id_global is set properly
        raise e

def compress_audio_step(audio_filename: str):
    global compressed_audio_filename
    logging.info(f"Compressing audio: {audio_filename}")
    try:
        compressed_audio_filename = ".".join(audio_filename.split(".")[:-1]) + ".mp3"
        audio = AudioSegment.from_file(audio_filename)
        audio.export(compressed_audio_filename, format="mp3", bitrate="32k")
        logging.info(f"Audio compressed and saved as: {compressed_audio_filename}")
        return compressed_audio_filename
    except Exception as e:
        logging.error(f"Error compressing audio: {e}")
        update_failed_status(video_id_global)
        raise e
    
# Function to get audio segments using langchain-openai
def get_segments(compressed_audio_filename):
    global segments
    logging.info(f"Getting audio segments from: {compressed_audio_filename}")

    try:
        if not segments:
            # langchain-openai Setup
            openai_api_key = os.getenv("OPENAI_API_KEY")  # Ensure your OpenAI API key is set as an environment variable
            
            # Initialize OpenAI API with langchain
            openai = OpenAI(api_key=openai_api_key)
            
            # OpenAI Whisper for audio transcription
            with open(compressed_audio_filename, "rb") as audio_file:
                response = openai.audio.transcriptions.create(
                    model="whisper-1",  # Whisper model for transcription
                    file=audio_file
                )
            
            # Assuming the response contains the transcription and segments
            print("Printing response")
            
            transcription_text = response.text
            print(transcription_text)
            segments=transcription_text
        
        logging.info(f"Got {len(segments)} segments")
        return segments

    except Exception as e:
        logging.error(f"Error occurred while processing audio: {str(e)}")
        return []
    
def update_transcript(video_id, transcript, table_name="videos", schema="public"):
    supabase_client = make_supabase_client(schema=schema)
    update_values = {"transcription": transcript}
    filter_conditions = {"video_id": video_id}
    supabase_client.table(table_name).update(update_values).match(filter_conditions).execute()

def transcribe_audio_step(compressed_audio_filename: str):
    global transcript
    logging.info(f"Transcribing audio from: {compressed_audio_filename}")
    try:
        transcript = ""
        segments = get_segments(compressed_audio_filename)
        # if segments:
        #     for segment in segments:
        #         transcript += segment['text']
        logging.info(f"Transcription completed: {transcript[:100]}...")
        update_transcript(video_id_global, segments)  # Update the transcript in the database
        return segments
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        update_failed_status(video_id_global)
        raise e

class TopicModel(BaseModel):
    starting_sentence: str = Field(..., description="The sentence in the transcript where the topic starts .This is for reference only")
    topic: str = Field(..., description="The topic covered in the video transcript.")
    explanation: str = Field(..., description="Provide a detail explanation of the topic along with all the subtopics explaination for better understanding.")
    examples: str = Field(..., description="Provide examples for the topic with code snippets to understand the concept better.")

class SummaryModel(BaseModel):
    language: str = Field(..., description="The programming language of the video transcript.")
    topics: list[TopicModel] = Field(..., description="The topics covered in the video transcript.")
    points_to_remember: list[str] = Field(..., description="Indexed important point to remember from the video transcript.")


def extract_alphanumeric(input_string):
    try:
        alphanumeric_string = re.sub(r'\W+', '', input_string)
        return alphanumeric_string
    except Exception as e:
        logging.error(f"Error extracting alphanumeric: {e}")
        update_failed_status(video_id_global)
        raise e

summary_parser = JsonOutputParser(pydantic_object=SummaryModel)
format_instructions = summary_parser.get_format_instructions()
summary_template = """
The following is transcription of the video:
{transcript}

{format_instructions}
    """

summary_prompt = PromptTemplate(
    template=summary_template,
    input_variables=["transcript"],
    partial_variables={"format_instructions": summary_parser.get_format_instructions()}
)

def update_data(video_id, summary, table_name="videos", schema="public"):
    supabase_client = make_supabase_client(schema=schema)
    update_values = {"summary": summary}
    filter_conditions = {"video_id": video_id}
    supabase_client.table(table_name).update(update_values).match(filter_conditions).execute()

def update_status(video_id, table_name="videos", schema="public"):
    supabase_client = make_supabase_client(schema=schema)
    update_values = {"status": "COMPLETED"}
    filter_conditions = {"video_id": video_id}
    supabase_client.table(table_name).update(update_values).match(filter_conditions).execute()

def save_summary_data(summary):
    update_data(video_id_global, summary)
    return summary




chain = (
      RunnableLambda(download_vimeo_video_step)
    | RunnableLambda(extract_audio_step)
    | RunnableLambda(compress_audio_step)
    | RunnableLambda(transcribe_audio_step)
    | summary_prompt
    | model
    | summary_parser
    | RunnableLambda(save_summary_data)
)