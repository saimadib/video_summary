import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from video_chain import chain as v_summary
from supabase import create_client, ClientOptions
import os
import logging
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Request model to take video URL
class VideoSummaryRequest(BaseModel):
    video_url: str

# Response model to return the summary
class VideoSummaryResponse(BaseModel):
    summary: str

# Initialize Supabase client globally
def make_supabase_client(schema="public"):
    opts = ClientOptions().replace(schema=schema)
    return create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY"),
        options=opts
    )

supabase_client = make_supabase_client()  # Initialize once for the app

@app.post("/process_video_summary")
async def process_video_summary_api(request: Request, parameter: VideoSummaryRequest):
    video_url = parameter.dict()["video_url"]

    print(video_url)
    
    try:
        # Store the video URL and get video_id
        video_id = store_video_url(video_url)

        print(video_id)

        # Start processing the video summary asynchronously
        asyncio.create_task(process_videos_sequentially([video_id]))

        return {"message": "Video summary processing has been triggered.", "video_id": video_id}
    except Exception as e:
        logging.error(f"Error in processing video summary: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Function to store video URL and return video ID
def store_video_url(video_url: str, table_name="videos", schema="public"):
    supabase_client = make_supabase_client(schema=schema)
    
    # Insert video URL into the database and get the video ID
    response = supabase_client.table(table_name).insert({"url": video_url, "status": "PENDING"}).execute()

    # Check if the response contains an error
    if hasattr(response, 'error') and response.error:
        logging.error(f"Error storing video URL: {response.error.message}")
        raise Exception(f"Failed to store video URL: {response.error.message}")
    
    # If insertion is successful, fetch the video_id from the response
    if hasattr(response, 'data') and response.data:
        video_id = response.data[0].get('video_id')  # Use 'id' instead of 'video_id' based on your response
        if video_id:
            logging.info(f"Stored video URL and assigned video_id: {video_id}")
            return video_id
        else:
            logging.error("Failed to store video URL: No id returned")
            raise Exception("Failed to store video URL: No id")
    else:
        logging.error("Failed to store video URL: No data returned")
        raise Exception("Failed to store video URL: No data returned")




# Function to process videos sequentially
async def process_videos_sequentially(video_ids):
    for video_id in video_ids:
        try:
            status = fetch_status(video_id)
            if status in ['IN_PROGRESS', 'COMPLETED']:
                continue
            if status in ['FAILED', 'PENDING']:
                logging.info(f"Starting generation for video {video_id}")
                await run_video_summary_chain(video_id)
        except Exception as e:
            logging.error(f"Error processing video {video_id}: {e}")

# Function to fetch video status
def fetch_status(video_id, table_name="videos", schema="public"):
    try:
        filter_conditions = {"video_id": video_id}
        response = supabase_client.table(table_name).select("status").match(filter_conditions).execute()
        
        if response.data and len(response.data) > 0:
            status = response.data[0]['status']
            logging.info(f"Fetched status for video_id {video_id}: {status}")
            return status
        else:
            logging.warning(f"No status found for video_id {video_id}")
            return None
    except Exception as e:
        logging.error(f"Error fetching status for video {video_id}: {e}")
        return None

# Function to update video status to "IN_PROGRESS"
def update_status(video_id, table_name="videos", schema="public"):
    try:
        update_values = {"status": "IN_PROGRESS"}
        filter_conditions = {"video_id": video_id}
        supabase_client.table(table_name).update(update_values).match(filter_conditions).execute()
        logging.info(f"Updated status to 'IN_PROGRESS' for video_id {video_id}")
    except Exception as e:
        logging.error(f"Error updating status for video {video_id}: {e}")

# Function to run the video summarization chain
async def run_video_summary_chain(video_id):
    try:
        update_status(video_id)  # Update status to IN_PROGRESS
        # Assuming video_summary.chain.invoke() is the asynchronous entry point
        await v_summary.ainvoke(video_id)
        logging.info(f"Video summarization completed for video_id {video_id}")
    except Exception as e:
        logging.error(f"Error in run_video_summary_chain for video {video_id}: {e}")

@app.get("/")
def read_root():
    return {"message": "Video summarization API is running!"}

if __name__ == "__main__":
    # This block will run the FastAPI app with Uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)
