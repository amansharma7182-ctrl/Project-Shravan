import asyncio
import json
import logging
import wave
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import whisper

# Configuration
SAMPLE_RATE = 8000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM = 2 bytes
RECORD_SECONDS = 15

TARGET_SAMPLES = SAMPLE_RATE * RECORD_SECONDS
TARGET_BYTES = TARGET_SAMPLES * SAMPLE_WIDTH

WHISPER_MODEL = "base"
TRANSCRIPTION_WORKERS = 2  # Concurrent Whisper workers


# File Locations
BASE_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = BASE_DIR / "recordings"
TRANSCRIPTIONS_FILE = BASE_DIR / "transcriptions.json"

RECORDINGS_DIR.mkdir(exist_ok=True)


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ASR-Server")


# Server State
class ServerState:
    def __init__(self):
        self.model: Any = None
        self.transcription_queue = None
        self.json_lock = None
        self.recording_id = 0


state = ServerState()


def get_recording_id() -> int:
    """Create and return a new recording ID."""
    state.recording_id += 1
    return state.recording_id


# Audio Input Processing
async def get_audio_chunk(websocket: WebSocket) -> bytes:
    """Receive one chunk of binary PCM audio from the WebSocket."""
    data = await websocket.receive_bytes()
    return data


async def get_audio_input(websocket: WebSocket) -> bytes:
    """Receive audio until TARGET_BYTES is reached (15 seconds max)."""
    audio = bytearray()

    while len(audio) < TARGET_BYTES:
        chunk = await get_audio_chunk(websocket)
        audio.extend(chunk)

        samples = len(audio) // SAMPLE_WIDTH
        percent = min(100, int((samples / TARGET_SAMPLES) * 100))

        # Log progress approximately every 10%
        if percent % 10 == 0:
            logger.info(f"Recording: {percent}% ({samples}/{TARGET_SAMPLES} samples)")

    return bytes(audio[:TARGET_BYTES])


# WAV File Handling
def save_audio(audio: bytes, recording_id: int) -> Path:
    """Save raw PCM audio as a WAV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"recording_{timestamp}_{recording_id}.wav"
    filepath = RECORDINGS_DIR / filename

    with wave.open(str(filepath), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio)

    return filepath


# Whisper Inference
def transcribe(filepath: str) -> dict:
    """Transcribe a WAV file using the loaded Whisper model."""
    result = state.model.transcribe(str(filepath), fp16=False, language="en")
    return {
        "text": result.get("text", "").strip(),
        "language": result.get("language", ""),
    }


# JSON Storage
def load_transcriptions() -> dict:
    """Load existing transcriptions or return default empty state."""
    if not TRANSCRIPTIONS_FILE.exists():
        return {"recordings": []}

    try:
        with open(TRANSCRIPTIONS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read transcriptions.json. Creating a new file.")
        return {"recordings": []}


def save_result_sync(result: dict) -> None:
    """Synchronously append a recording result to the JSON file."""
    data = load_transcriptions()
    data["recordings"].append(result)

    with open(TRANSCRIPTIONS_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


async def save_result(
    recording_id: int,
    timestamp: str,
    filepath: Path,
    transcription: dict,
) -> None:
    """Safely append transcription results using an async lock."""
    result = {
        "id": recording_id,
        "timestamp": timestamp,
        "audio_file": filepath.name,

        "sample_rate": SAMPLE_RATE,
        "duration_seconds": RECORD_SECONDS,
        "language": transcription["language"],
        "text": transcription["text"],
    }

    async with state.json_lock:
        await asyncio.to_thread(save_result_sync, result)


# Background Transcription Workers
async def transcription_worker(worker_id: int):
    """Worker task processing jobs from the queue via Whisper."""
    logger.info(f"Transcription worker {worker_id} started.")

    while True:
        job = await state.transcription_queue.get()

        try:
            recording_id = job["id"]
            filepath = Path(job["filepath"])
            timestamp = job["timestamp"]

            logger.info(f"Worker {worker_id}: Transcribing Recording #{recording_id}")

            # Offload CPU-heavy transcription off the main loop
            result = await asyncio.to_thread(transcribe, filepath)

            logger.info(f"Recording #{recording_id}: '{result['text']}'")
            await save_result(recording_id, timestamp, filepath, result)
            logger.info(f"Recording #{recording_id} completed.")

        except Exception:
            logger.exception(f"Worker {worker_id} failed.")
        finally:
            state.transcription_queue.task_done()


async def run_parallel_transcription():
    """Spawn the worker pool tasks."""
    workers = []
    for worker_id in range(TRANSCRIPTION_WORKERS):
        worker = asyncio.create_task(transcription_worker(worker_id + 1))
        workers.append(worker)
    return workers


# Audio Connection Handling
async def handle_audio_connection(websocket: WebSocket):
    """Manage a single ESP32 WebSocket audio stream session."""
    recording_id = get_recording_id()
    timestamp = datetime.now().isoformat()

    logger.info(f"ESP32 connected. Recording #{recording_id}")

    try:
        # 1. Receive Audio
        audio = await get_audio_input(websocket)
        logger.info(f"Recording #{recording_id}: Audio received.")

        # 2. Save Audio
        filepath = await asyncio.to_thread(save_audio, audio, recording_id)
        logger.info(f"Recording #{recording_id}: Saved as {filepath.name}")

        # 3. Queue for Transcription
        await state.transcription_queue.put(
            {
                "id": recording_id,
                "filepath": str(filepath),
                "timestamp": timestamp,
            }
        )
        logger.info(f"Recording #{recording_id}: Added to transcription queue.")

    except WebSocketDisconnect:
        logger.warning(f"ESP32 disconnected during Recording #{recording_id}")
    except Exception:
        logger.exception(f"Error handling Recording #{recording_id}")


# Lifespan and Application Initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("================================")
    logger.info("       ASR SERVER STARTING      ")
    logger.info("================================")

    # Initialize async objects inside the running loop
    state.transcription_queue = asyncio.Queue()
    state.json_lock = asyncio.Lock()

    # Load Whisper Model
    logger.info(f"Loading Whisper '{WHISPER_MODEL}'...")
    state.model = await asyncio.to_thread(whisper.load_model, WHISPER_MODEL)
    logger.info("Whisper model loaded.")

    # Start Workers
    workers = await run_parallel_transcription()
    logger.info(f"{TRANSCRIPTION_WORKERS} transcription workers running.")

    logger.info("================================")
    logger.info("          SERVER READY          ")
    logger.info("================================")

    yield

    # Shutdown Procedures
    logger.info("Shutting down workers...")
    for worker in workers:
        worker.cancel()

    await asyncio.gather(*workers, return_exceptions=True)
    logger.info("ASR server stopped.")


# FastAPI Initialization
app = FastAPI(lifespan=lifespan)


# WebSocket Route
@app.websocket("/audio")
async def audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await handle_audio_connection(websocket)
    finally:
        logger.info("ESP32 audio session closed.")
        


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
