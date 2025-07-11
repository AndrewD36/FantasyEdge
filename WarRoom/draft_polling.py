import asyncio
from sleeper_access import get_draft_picks
from redis_client import redis_conn
from draft_analyzer import analyze_all_teams

async def poll_draft_picks(draft_id: str):
    stored_picks_key = f'draft:{draft_id}:picks'
    seen_pick_id = set()

    team_ids = await redis_conn.smembers(f'draft:{draft_id}:teams')

    while True:
        picks = await get_draft_picks(draft_id)

        for pick in picks:
            pick_id = f'{pick['round']}-{pick['pick_no']}'

            if not await redis_conn.sismember(stored_picks_key, pick_id):
                await redis_conn.rpush(f'{stored_picks_key}:data', str(pick))
                await redis_conn.sadd(stored_picks_key, pick_id)

                await update_team_roster(pick)

                await analyze_all_teams(draft_id, list(team_ids))

                print(f'[+] New Pick: {pick['metadata']['first_name']} {pick['metadata']['last_name']}')

        await asyncio.sleep(3)

async def update_team_roster(pick: dict):
    team_id = str(pick["roster_id"])
    player_id = str(pick["player_id"])
    metadata = pick.get(metadata, {})

    if not metadata:
        print(f'[!] Missing metadata for pick {pick}')
        return
    
    await redis_conn.sadd(f'team:{team_id}:roster', player_id)
    await redis_conn.hset(f'player:{player_id}', mapping={
        "name": f'{metadata.get('first_name', '')} {metadata.get('last_name', '')}'.strip(),
        "position": metadata.get('position', ''),
        "team": metadata.get('team', ''),
    })

    await redis_conn.sadd('picked_players', player_id)

