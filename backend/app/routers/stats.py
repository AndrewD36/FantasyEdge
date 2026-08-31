from typing import Literal, Optional
 
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, desc, asc
from sqlalchemy.orm import Session
 
from database import get_db
from models import Player, PlayerStat
 
router = APIRouter(prefix="/stats", tags=["stats"])
 
# Whitelist of sortable columns — never interpolate the query param straight
# into getattr() against arbitrary model attributes.
SORTABLE_FIELDS = {
    "fantasy_points_ppr",
    "fantasy_points",
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "receptions",
    "targets",
}
SortField = Literal[
    "fantasy_points_ppr", "fantasy_points", "passing_yards", "passing_tds",
    "rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds",
    "receptions", "targets",
]
 
 
class LeaderboardEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)
 
    player_id: str
    full_name: str
    position: Optional[str]
    team_abbr: Optional[str]
    season: int
    week: int
    fantasy_points: Optional[float]
    fantasy_points_ppr: Optional[float]
    passing_yards: float
    passing_tds: int
    rushing_yards: float
    rushing_tds: int
    receiving_yards: float
    receiving_tds: int
    receptions: int
    targets: int
 
 
@router.get("/leaderboard", response_model=list[LeaderboardEntry])
def get_leaderboard(
    season: int = Query(..., description="Required — this joins across all players for one season"),
    week: Optional[int] = Query(None, ge=1, le=22, description="Omit to rank season-to-date totals... (see note)"),
    position: Optional[str] = Query(None, description="e.g. RB, WR"),
    team: Optional[str] = Query(None, description="Team abbreviation"),
    season_type: str = Query("REG", pattern="^(REG|POST|PRE)$"),
    sort_by: SortField = Query("fantasy_points_ppr"),
    order: Literal["asc", "desc"] = "desc",
    limit: int = Query(25, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Cross-player ranking for a season/week — the 'who had the best game'
    query. Doesn't belong under /players since it spans all of them.
 
    Note: this ranks individual game rows, not season totals. If week is
    omitted, each player's every game that season is a separate ranked row —
    for season aggregates you'd want a separate /stats/season-totals
    endpoint doing a SUM/GROUP BY, which isn't implemented here yet."""
    stmt = (
        select(PlayerStat, Player.full_name, Player.position)
        .join(Player, Player.player_id == PlayerStat.player_id)
        .where(PlayerStat.season == season, PlayerStat.season_type == season_type)
    )
    if week:
        stmt = stmt.where(PlayerStat.week == week)
    if position:
        stmt = stmt.where(Player.position == position.upper())
    if team:
        stmt = stmt.where(PlayerStat.team_abbr == team.upper())
 
    sort_col = getattr(PlayerStat, sort_by)  # safe: sort_by is Literal-validated by FastAPI
    stmt = stmt.order_by(desc(sort_col) if order == "desc" else asc(sort_col))
    stmt = stmt.limit(limit)
 
    rows = db.execute(stmt).all()
    return [
        LeaderboardEntry(
            player_id=stat.player_id,
            full_name=full_name,
            position=position,
            team_abbr=stat.team_abbr,
            season=stat.season,
            week=stat.week,
            fantasy_points=stat.fantasy_points,
            fantasy_points_ppr=stat.fantasy_points_ppr,
            passing_yards=stat.passing_yards,
            passing_tds=stat.passing_tds,
            rushing_yards=stat.rushing_yards,
            rushing_tds=stat.rushing_tds,
            receiving_yards=stat.receiving_yards,
            receiving_tds=stat.receiving_tds,
            receptions=stat.receptions,
            targets=stat.targets,
        )
        for stat, full_name, position in rows
    ]