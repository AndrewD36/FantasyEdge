from fastapi import APIRouter

from app.database import get_db_connection

router = APIRouter(
    prefix="/stats",
    tags=["stats"],
)


@router.get("/receiving")
def get_receiving_stats(
    season: int,
    week: int | None = None,
):
    connection = get_db_connection()

    if week is not None:
        rows = connection.execute(
            """
            SELECT
                player_id,
                player_display_name,
                team,
                position,
                week,
                targets,
                receptions,
                receiving_yards,
                receiving_tds
            FROM player_stats
            WHERE season = ?
              AND week = ?
            ORDER BY receiving_yards DESC
            """,
            (season, week),
        ).fetchall()
    else:
        rows = connection.execute(
            """
            SELECT
                player_id,
                player_display_name,
                team,
                position,
                SUM(targets) AS targets,
                SUM(receptions) AS receptions,
                SUM(receiving_yards) AS receiving_yards,
                SUM(receiving_tds) AS receiving_tds
            FROM player_stats
            WHERE season = ?
            GROUP BY
                player_id,
                player_display_name,
                team,
                position
            ORDER BY receiving_yards DESC
            """,
            (season,),
        ).fetchall()

    connection.close()

    return [dict(row) for row in rows]