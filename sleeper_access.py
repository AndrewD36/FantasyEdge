import httpx

BASE_URL = "https://api.sleeper.app/v1"

async def get_user_data(username: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f'{BASE_URL}/user/{username}')
        return response.json()
    
async def get_user_leagues(user_id: str, season: int):
    async with httpx.AsyncClient() as client:
        response = await client.get(f'{BASE_URL}/user/{user_id}/leagues/nfl/{season}')
        return response.json()
    
async def get_league_drafts(league_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f'{BASE_URL}/league/{league_id}/drafts')
        return response.json()
    
async def get_draft_picks(draft_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f'{BASE_URL}/draft/{draft_id}/picks')
        return response.json()