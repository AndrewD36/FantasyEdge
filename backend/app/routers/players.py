from typing import Optional
 
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session
 
from database import get_db
from models import Player, PlayerStat, PlayerTeamSeason
from schemas import PlayerRead, PlayerStatRead, PlayerWithStatsRead
 
router = APIRouter(prefix="/players", tags=["players"])

@router.get("", response_model=list[PlayerRead])
def list_players(
    position: Optional[str] = Query(None, description="e.g. QB, RB, WR, TE"),
    team: Optional[str] = Query(None, description="Team abbreviation, e.g. SF"),
    season: Optional[int] = Query(
        None, description="Restrict the team filter to this season (defaults to any season on record)"
    ),
    search: Optional[str] = Query(None, min_length=2, description="Case-insensitive substring match on player name"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Browse/search the player dimension. Every filter here narrows *who*
    you get back — for filtering by performance, use /stats/leaderboard."""
    stmt = select(Player)
 
    if position:
        stmt = stmt.where(Player.position == position.upper())
    if search:
        stmt = stmt.where(Player.full_name.ilike(f"%{search}%"))
    if team:
        team_player_ids = select(PlayerTeamSeason.player_id).where(
            PlayerTeamSeason.team_abbr == team.upper()
        )
        if season:
            team_player_ids = team_player_ids.where(PlayerTeamSeason.season == season)
        stmt = stmt.where(Player.player_id.in_(team_player_ids))
 
    stmt = stmt.order_by(Player.full_name).offset(offset).limit(limit)
    return db.execute(stmt).scalars().all()
 
 
@router.get("/{player_id}", response_model=PlayerWithStatsRead)
def get_player(
    player_id: str,
    include_stats: bool = Query(False, description="Attach this player's stat log to the response"),
    season: Optional[int] = Query(None, description="If include_stats, restrict the log to this season"),
    db: Session = Depends(get_db),
):
    player = db.get(Player, player_id)
    if not player:
        raise HTTPException(status_code=404, detail=f"No player with id '{player_id}'")
 
    stats: list[PlayerStat] = []
    if include_stats:
        stmt = select(PlayerStat).where(PlayerStat.player_id == player_id)
        if season:
            stmt = stmt.where(PlayerStat.season == season)
        stmt = stmt.order_by(PlayerStat.season, PlayerStat.week)
        stats = db.execute(stmt).scalars().all()
 
    # Build explicitly rather than returning the ORM object directly -
    # letting Pydantic walk `player.stats` would lazy-load the *entire*
    # relationship regardless of include_stats, silently ignoring the flag.
    return PlayerWithStatsRead(
        **PlayerRead.model_validate(player).model_dump(),
        stats=[PlayerStatRead.model_validate(s) for s in stats],
    )
 
 
@router.get("/{player_id}/stats", response_model=list[PlayerStatRead])
def get_player_stats(
    player_id: str,
    season: int = Query(..., description="Required — the stats table is large, so a season scopes the query"),
    week: Optional[int] = Query(None, ge=1, le=22),
    season_type: str = Query("REG", pattern="^(REG|POST|PRE)$"),
    db: Session = Depends(get_db),
):
    """A single player's game log — the natural sub-resource of a player."""
    if not db.get(Player, player_id):
        raise HTTPException(status_code=404, detail=f"No player with id '{player_id}'")
 
    stmt = select(PlayerStat).where(
        PlayerStat.player_id == player_id,
        PlayerStat.season == season,
        PlayerStat.season_type == season_type,
    )
    if week:
        stmt = stmt.where(PlayerStat.week == week)
    stmt = stmt.order_by(PlayerStat.week)
    return db.execute(stmt).scalars().all()