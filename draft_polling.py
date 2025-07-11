import asyncio
from sleeper_access import get_draft_picks
from redis_client import redis_conn

async def poll_draft_picks(draft_id: str):
    stored_picks_key = f'draft:{draft_id}:picks'
    seen_pick_id = set()

    while True:
        picks = await get_draft_picks(draft_id)

        for pick in picks:
            pick_id = f'{pick['round']}-{pick['pick_no']}'

            if not await redis_conn.sismember(stored_picks_key, pick_id):
                await redis_conn.rpush(f'{stored_picks_key}:data', str(pick))
                await redis_conn.sadd(stored_picks_key, pick_id)

                print(f'[+] New Pick: {pick['metadata']['first_name']} {pick['metadata']['last_name']}')

        await asyncio.sleep(3)