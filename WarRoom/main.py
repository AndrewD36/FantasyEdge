from fastapi import FastAPI, WebSocket
from sleeper_access import get_user_data, get_user_leagues, get_league_drafts, get_draft_picks
from draft_polling import poll_draft_picks
from WarRoom.draft_analyzer import test_model
from datetime import datetime as dt
import asyncio

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Fantasy Edge, your personal fantasy football companion app!"}

@app.get("/user/{username}")
async def get_user(username: str):
    user_data = await get_user_data(username)
    return user_data

@app.get("/")
async def get_leagues(username: str, season: int = dt.now().year):
    user_data = await get_user_data(username)
    user_id = user_data["user_id"]
    leagues = await get_user_leagues(user_id, season)
    return leagues

@app.get("/drafts/{league_id}")
async def get_drafts(league_id: str):
    drafts = await get_league_drafts(league_id)
    return drafts

@app.get("/draft-picks/{draft_id}")
async def get_picks(draft_id: str):
    picks = await get_draft_picks(draft_id)
    return picks

@app.on_event("startup")
async def draft_polling():
    draft_id = 'YOUR_DRAFT_ID'
    asyncio.create_task(poll_draft_picks(draft_id))

# MOVE TO THIS:
async def lifespan(app: FastAPI):
    pass

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
