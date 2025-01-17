from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import httpx
import json
import os
import logging
from dotenv import load_dotenv
import datetime
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "text/event-stream"]
)

# Store the text state
class TextState:
    def __init__(self):
        self.current_text = None
        self.last_updated = datetime.datetime.now()

    def set_text(self, text):
        self.current_text = text
        self.last_updated = datetime.datetime.now()

    def clear_text(self):
        self.current_text = None

text_state = TextState()

class TTSRequest(BaseModel):
    input: dict
    voice: dict
    audioConfig: dict

class ReceiveText(BaseModel):
    text: str

async def get_access_token():
    """Get access token from API key"""
    api_key = os.getenv("GOOGLE_TTS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Google API key not configured")
    
    url = f"https://eu-texttospeech.googleapis.com/v1beta1/text:synthesize?key={api_key}"
    return url

async def event_generator():
    while True:
        if text_state.current_text:
            # Get the current text and clear it immediately
            text_to_send = text_state.current_text
            text_state.clear_text()
            
            yield {
                "event": "message",
                "data": {
                    "text": text_to_send,
                    "timestamp": text_state.last_updated.isoformat()
                }
            }
        await asyncio.sleep(0.5)  # Check for updates every half second

@app.post("/receive")
async def receive_text(text_data: ReceiveText):
    """Receive text from external source and update current_text"""
    if text_data.text:
        text_state.set_text(text_data.text)
        logger.info(f"Received new text: {text_data.text}")
        return {"status": "success", "text": text_data.text}
    return {"status": "skipped", "message": "No text provided"}

@app.get("/stream")
async def message_stream():
    return EventSourceResponse(event_generator())

@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    try:
        # Get URL with API key
        url = await get_access_token()
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Prepare request payload
        payload = {
            "input": request.input,
            "voice": request.voice,
            "audioConfig": request.audioConfig
        }

        # Make request to Google TTS API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=30.0
            )
            
            # Check for errors
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Google TTS API error: {response.text}"
                )
            
            return response.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to Google TTS API timed out")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"HTTP error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "current_text": text_state.current_text,
        "last_updated": text_state.last_updated.isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or default to 8000
    port = int(os.getenv("PORT", 10000))
    
    logger.info(f"Starting server on port {port}...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Enable auto-reload during development

        #pip install sse-starlette
        log_level="info"
    )
