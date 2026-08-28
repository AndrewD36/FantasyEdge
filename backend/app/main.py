from fastapi import FastAPI

from app.routers.stats import router as stats_router

app = FastAPI(
    title="NFL Stats API"
)

app.include_router(stats_router)


@app.get("/")
def root():
    return {"message": "NFL Stats API"}
