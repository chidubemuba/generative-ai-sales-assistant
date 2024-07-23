
from fastapi import APIRouter, WebSocket
import asyncio

router = APIRouter()

@router.get("/recommendation-response")
async def get_recommendation():
    return {"recommendation": ['this is a recommendation']}

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Send a message every 20 seconds
        await websocket.send_text("This is a periodic message from the server.")
        await asyncio.sleep(20)  # Sleep for 20 seconds