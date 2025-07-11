import json
from redis_client import redis_conn

CORE_POSITIONS = ["QB", "RB", "WR", "TE"]

async def analyze_all_teams(draft_id: str, teams_ids: list[str]):
    for team_id in teams_ids:
        await analyze_team_needs(draft_id, team_id)

async def analyze_team_needs(draft_id, team_id):
    roster_key = f'team:{team_id}:roster'
    suggestions_key = f'team:{team_id}:suggestions'

    roster = await redis_conn.smembers(roster_key)

    position_counts = {"pos": 0 for position in CORE_POSITIONS}

    for player_id in roster:
        pos = await redis_conn.hget(f'player:{player_id}', "position")
        if pos in position_counts:
            position_counts[pos] += 1

    position_priority = sorted(CORE_POSITIONS, key=lambda x: position_counts[x])

    suggestions = []

    for pos in position_priority:
        available_key = f'available_players{pos}'
        available_ids = await redis_conn.lrange(available_key, 0, 10)

        for pid in available_ids:
            if pid not in roster:
                pdata = await redis_conn.hgetall(f'player{pid}')

                if not pdata: continue

                suggestions.append({"player_id": pid,
                                    "name": pdata.get("name"),
                                    "position": pos,
                                    "reason": f'Only {position_counts[pos]} {pos}s on roster'
                                    })
                break

        if len(suggestions) >= 3:
            break

    await redis_conn.set(suggestions_key, json.dumps(suggestions))