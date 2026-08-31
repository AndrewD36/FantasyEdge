from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, or_
from sqlalchemy.orm import Session

from database import get_db
from models import Game, Team
from schemas import GameRead

router = APIRouter(prefix="/games", tags=["games"])


def _games_query(season: int, week: Optional[int], team: Optional[str]):
    """Shared filter logic - teams.py's /schedule sub-resource is just this
    same query pre-scoped to one team, so it reuses this rather than
    duplicating the where-clauses."""
    stmt = select(Game).where(Game.season == season)
    if week:
        stmt = stmt.where(Game.week == week)
    if team:
        team = team.upper()
        stmt = stmt.where(or_(Game.home_team == team, Game.away_team == team))
    return stmt.order_by(Game.week, Game.game_id)


@router.get("", response_model=list[GameRead])
def list_games(
    season: int = Query(..., description="Required — games are naturally scoped by season"),
    week: Optional[int] = Query(None, ge=1, le=22),
    team: Optional[str] = Query(None, description="Team abbreviation; matches either home or away"),
    db: Session = Depends(get_db),
):
    return db.execute(_games_query(season, week, team)).scalars().all()


@router.get("/{game_id}", response_model=GameRead)
def get_game(game_id: str, db: Session = Depends(get_db)):
    game = db.get(Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail=f"No game with id '{game_id}'")
    return game