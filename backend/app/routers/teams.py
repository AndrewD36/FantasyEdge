from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import get_db
from models import Player, PlayerTeamSeason, Team
from schemas import TeamSchema, GameRead
from routers.games import _games_query

router = APIRouter(prefix="/teams", tags=["teams"])


class RosterEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    player_id: str
    full_name: str
    position: Optional[str] = None
    week: int


@router.get("", response_model=list[TeamSchema])
def list_teams(db: Session = Depends(get_db)):
    return db.execute(select(Team).order_by(Team.team_abbr)).scalars().all()


@router.get("/{team_abbr}", response_model=TeamSchema)
def get_team(team_abbr: str, db: Session = Depends(get_db)):
    team = db.get(Team, team_abbr.upper())
    if not team:
        raise HTTPException(status_code=404, detail=f"No team '{team_abbr}'")
    return team


@router.get("/{team_abbr}/roster", response_model=list[RosterEntry])
def get_team_roster(
    team_abbr: str,
    season: int = Query(..., description="Required — rosters are tracked per season"),
    week: Optional[int] = Query(None, ge=0, le=22, description="Specific week; omit for the season-level roster"),
    db: Session = Depends(get_db),
):
    """Sub-resource: which players were on this team, for a given season/week.
    Reads from the player_team_seasons bridge, so it correctly reflects
    mid-season trades rather than a single 'current team' field."""
    if not db.get(Team, team_abbr.upper()):
        raise HTTPException(status_code=404, detail=f"No team '{team_abbr}'")

    stmt = (
        select(PlayerTeamSeason, Player.full_name, Player.position)
        .join(Player, Player.player_id == PlayerTeamSeason.player_id)
        .where(
            PlayerTeamSeason.team_abbr == team_abbr.upper(),
            PlayerTeamSeason.season == season,
        )
    )
    if week is not None:
        stmt = stmt.where(PlayerTeamSeason.week == week)
    stmt = stmt.order_by(Player.position, Player.full_name)

    rows = db.execute(stmt).all()
    return [
        RosterEntry(
            player_id=pts.player_id,
            full_name=full_name,
            position=position,
            week=pts.week,
        )
        for pts, full_name, position in rows
    ]

@router.get("/{team_abbr}/schedule", response_model=list[GameRead])
def get_team_schedule(
    team_abbr: str,
    season: int = Query(..., description="Required — schedules are tracked per season"),
    week: Optional[int] = Query(None, ge=1, le=22),
    db: Session = Depends(get_db),
):
    """Thin wrapper around GET /games?team=X&season=Y - same query, just
    addressed as a sub-resource of the team rather than a filtered list."""
    if not db.get(Team, team_abbr.upper()):
        raise HTTPException(status_code=404, detail=f"No team '{team_abbr}'")
    return db.execute(_games_query(season, week, team_abbr)).scalars().all()