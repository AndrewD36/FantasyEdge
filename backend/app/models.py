from sqlalchemy import (ForeignKey, Index, String, Integer, Float, Date, DateTime, UniqueConstraint, func)
from sqlalchemy.orm import ( DeclarativeBase, Mapped, mapped_column, relationship)
from datetime import date, datetime
from typing import Optional

class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    team_abbr: Mapped[str] = mapped_column(String(3), primary_key=True)
    team_name: Mapped[str] = mapped_column(String(64), nullable=False)
    conference: Mapped[Optional[str]] = mapped_column(String(3))
    division: Mapped[Optional[str]] = mapped_column(String(8))

    stats: Mapped[list["PlayerStat"]] = relationship(
        back_populates="team", foreign_keys="PlayerStat.team_abbr"
    )


class Player(Base):
    __tablename__ = "players"
 
    player_id: Mapped[str] = mapped_column(String(16), primary_key=True)  # gsis_id
    full_name: Mapped[str] = mapped_column(String(128), nullable=False)
    first_name: Mapped[Optional[str]] = mapped_column(String(64))
    last_name: Mapped[Optional[str]] = mapped_column(String(64))
    position: Mapped[Optional[str]] = mapped_column(String(4))
    birth_date: Mapped[Optional[date]] = mapped_column(Date)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    weight: Mapped[Optional[int]] = mapped_column(Integer)
    college: Mapped[Optional[str]] = mapped_column(String(64))
    draft_year: Mapped[Optional[int]] = mapped_column(Integer)
    draft_round: Mapped[Optional[int]] = mapped_column(Integer)
    draft_pick: Mapped[Optional[int]] = mapped_column(Integer)
    rookie_season: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[Optional[str]] = mapped_column(String(16))
 
    # cross-source IDs, useful once you wire in Sleeper live data
    espn_id: Mapped[Optional[str]] = mapped_column(String(16))
    sleeper_id: Mapped[Optional[str]] = mapped_column(String(16), index=True)
    yahoo_id: Mapped[Optional[str]] = mapped_column(String(16))
 
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
 
    stats: Mapped[list["PlayerStat"]] = relationship(
        back_populates="player", foreign_keys="PlayerStat.player_id"
    )
    team_seasons: Mapped[list["PlayerTeamSeason"]] = relationship(
        back_populates="player"
    )

class PlayerTeamSeason(Base):
    """Bridge table: which team a player was on, by season/week.
    Handles in-season trades without corrupting the player dimension."""
 
    __tablename__ = "player_team_seasons"
 
    player_id: Mapped[str] = mapped_column(
        ForeignKey("players.player_id"), primary_key=True
    )
    season: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_abbr: Mapped[str] = mapped_column(
        ForeignKey("teams.team_abbr"), primary_key=True
    )
    week: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True, default=0)
 
    player: Mapped["Player"] = relationship(back_populates="team_seasons")
    team: Mapped["Team"] = relationship()
 
 
class PlayerStat(Base):
    """Fact table: one row per player per game. Wide, not EAV —
    mirrors nflreadpy.load_player_stats() grain almost 1:1."""
 
    __tablename__ = "player_stats"
    __table_args__ = (
        UniqueConstraint(
            "player_id", "season", "week", "season_type", name="uq_player_game"
        ),
        Index("idx_stats_player_season", "player_id", "season"),
        Index("idx_stats_season_week", "season", "week"),
    )
 
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
 
    player_id: Mapped[str] = mapped_column(ForeignKey("players.player_id"))
    season: Mapped[int] = mapped_column(Integer)
    week: Mapped[int] = mapped_column(Integer)
    season_type: Mapped[str] = mapped_column(String(4), default="REG")
 
    team_abbr: Mapped[Optional[str]] = mapped_column(ForeignKey("teams.team_abbr"))
    opponent_abbr: Mapped[Optional[str]] = mapped_column(String(3))
 
    # passing
    completions: Mapped[int] = mapped_column(Integer, default=0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    passing_yards: Mapped[float] = mapped_column(Float, default=0)
    passing_tds: Mapped[int] = mapped_column(Integer, default=0)
    interceptions: Mapped[int] = mapped_column(Integer, default=0)
    sacks: Mapped[int] = mapped_column(Integer, default=0)
    sack_yards: Mapped[float] = mapped_column(Float, default=0)
 
    # rushing
    carries: Mapped[int] = mapped_column(Integer, default=0)
    rushing_yards: Mapped[float] = mapped_column(Float, default=0)
    rushing_tds: Mapped[int] = mapped_column(Integer, default=0)
    rushing_fumbles: Mapped[int] = mapped_column(Integer, default=0)
 
    # receiving
    targets: Mapped[int] = mapped_column(Integer, default=0)
    receptions: Mapped[int] = mapped_column(Integer, default=0)
    receiving_yards: Mapped[float] = mapped_column(Float, default=0)
    receiving_tds: Mapped[int] = mapped_column(Integer, default=0)
    receiving_fumbles: Mapped[int] = mapped_column(Integer, default=0)
 
    # fantasy (nflreadpy computes these already; don't recompute yourself)
    fantasy_points: Mapped[Optional[float]] = mapped_column(Float)
    fantasy_points_ppr: Mapped[Optional[float]] = mapped_column(Float)
 
    player: Mapped["Player"] = relationship(back_populates="stats")
    team: Mapped[Optional["Team"]] = relationship(back_populates="stats")