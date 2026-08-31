from fastapi import FastAPI
 
from routers import players, teams, stats, games

from fastapi.middleware.cors import CORSMiddleware
 
app = FastAPI(
    title="FantasyEdge Data API",
    description="Read API over nflreadpy-sourced data.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
app.include_router(players.router)
app.include_router(teams.router)
app.include_router(stats.router)
app.include_router(games.router)
 
 
@app.get("/health")
def health():
    return {"status": "ok"}