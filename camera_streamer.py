import os
import asyncio
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from reachy_mini import ReachyMini

# Import the existing CameraWorker from the app
from reachy_mini_conversation_app.camera_worker import CameraWorker

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("camera_streamer")

app = FastAPI()

# Initialize robot and camera worker as globals for the app lifecycle
robot = None
camera_worker = None

@app.on_event("startup")
async def startup_event():
    global robot, camera_worker
    logger.info("Connecting to Reachy Mini...")
    robot = ReachyMini()
    logger.info("Starting Camera Worker...")
    camera_worker = CameraWorker(robot)
    camera_worker.start()
    logger.info("Camera streamer ready at http://0.0.0.0:8000")

@app.on_event("shutdown")
async def shutdown_event():
    if camera_worker:
        camera_worker.stop()
    if robot:
        robot.client.disconnect()
    logger.info("Shutdown complete.")

@app.get("/video_feed")
async def video_feed():
    """MJPEG streaming endpoint used by index.html."""
    async def generate():
        while True:
            if camera_worker:
                jpeg = camera_worker.get_latest_jpeg()
                if jpeg:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            await asyncio.sleep(0.04) # ~25 FPS

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# Mount static files (images, css, js, signs) at the root
# We mount this LAST so that /video_feed takes precedence
STATIC_DIR = os.path.join(BASE_DIR, "src", "reachy_mini_conversation_app", "static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    # Run the server on all interfaces so it's accessible on your network
    uvicorn.run(app, host="0.0.0.0", port=8000)
