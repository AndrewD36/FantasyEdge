from fastapi import FastAPI
 
from routers import players, teams, stats
 
app = FastAPI(
    title="FantasyEdge Data API",
    description="Read API over nflreadpy-sourced player, team, and stat data.",
    version="0.1.0",
)
 
app.include_router(players.router)
app.include_router(teams.router)
app.include_router(stats.router)
 
 
@app.get("/health")
def health():
    return {"status": "ok"}