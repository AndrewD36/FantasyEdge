from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from redis_client import redis_conn
import json
import asyncio

router = APIRouter()

@router.websocket("/ws/draft/{draft_id}")
async def ws_draft(websocket: WebSocket, draft_id: str):
    await websocket.accept()

    try:
        while True:
            teams = await redis_conn.smembers(f'draft:{draft_id}:teams')
            all_suggestions = {}

            for tid in teams:
                raw = await redis_conn.get(f'team:{tid}:suggestions')
                if raw:
                    all_suggestions[tid] = json.loads(raw)

            await websocket.send_json({
                "type": "suggestions",
                "data": all_suggestions
            })

            await asyncio.sleep(3)

    except WebSocketDisconnect:
        print("Client Disconnect")