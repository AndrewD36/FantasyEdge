from fastapi import FastAPI
 
from routers import players, teams, stats, games
 
app = FastAPI(
    title="FantasyEdge Data API",
    description="Read API over nflreadpy-sourced data.",
    version="0.2.0",
)
 
app.include_router(players.router)
app.include_router(teams.router)
app.include_router(stats.router)
app.include_router(games.router)
 
 
@app.get("/health")
def health():
    return {"status": "ok"}