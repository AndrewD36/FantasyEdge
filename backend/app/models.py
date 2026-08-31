from sqlalchemy import (ForeignKey, Index, String, Integer, Float, Date, DateTime, Boolean, UniqueConstraint, func)
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
    snap_counts: Mapped[list["SnapCount"]] = relationship(back_populates="player")
    ngs_passing: Mapped[list["NgsPassing"]] = relationship(back_populates="player")
    ngs_receiving: Mapped[list["NgsReceiving"]] = relationship(back_populates="player")
    ngs_rushing: Mapped[list["NgsRushing"]] = relationship(back_populates="player")

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


class Game(Base):
    """One row per game, from nflreadpy.load_schedules(). Includes Vegas
    lines (spread/total) — genuinely useful for fantasy projections since
    implied team totals predict scoring environment better than raw stats
    alone. Not just a schedule lookup table."""

    __tablename__ = "games"

    game_id: Mapped[str] = mapped_column(String(20), primary_key=True)  # e.g. "2023_01_DET_KC"
    season: Mapped[int] = mapped_column(Integer, index=True)
    week: Mapped[int] = mapped_column(Integer)
    game_type: Mapped[str] = mapped_column(String(4))  # REG, WC, DIV, CON, SB
    gameday: Mapped[Optional[date]] = mapped_column(Date)
    weekday: Mapped[Optional[str]] = mapped_column(String(16))
    gametime: Mapped[Optional[str]] = mapped_column(String(8))

    away_team: Mapped[str] = mapped_column(ForeignKey("teams.team_abbr"))
    home_team: Mapped[str] = mapped_column(ForeignKey("teams.team_abbr"))
    away_score: Mapped[Optional[int]] = mapped_column(Integer)
    home_score: Mapped[Optional[int]] = mapped_column(Integer)
    result: Mapped[Optional[int]] = mapped_column(Integer)  # home - away margin
    total: Mapped[Optional[int]] = mapped_column(Integer)  # combined points scored
    overtime: Mapped[bool] = mapped_column(Boolean, default=False)
    div_game: Mapped[bool] = mapped_column(Boolean, default=False)

    roof: Mapped[Optional[str]] = mapped_column(String(16))
    surface: Mapped[Optional[str]] = mapped_column(String(16))
    temp: Mapped[Optional[int]] = mapped_column(Integer)
    wind: Mapped[Optional[int]] = mapped_column(Integer)

    away_qb_id: Mapped[Optional[str]] = mapped_column(ForeignKey("players.player_id"))
    home_qb_id: Mapped[Optional[str]] = mapped_column(ForeignKey("players.player_id"))
    away_qb_name: Mapped[Optional[str]] = mapped_column(String(64))
    home_qb_name: Mapped[Optional[str]] = mapped_column(String(64))
    away_coach: Mapped[Optional[str]] = mapped_column(String(64))
    home_coach: Mapped[Optional[str]] = mapped_column(String(64))
    referee: Mapped[Optional[str]] = mapped_column(String(64))
    stadium: Mapped[Optional[str]] = mapped_column(String(128))

    spread_line: Mapped[Optional[float]] = mapped_column(Float)  # negative = home favored
    total_line: Mapped[Optional[float]] = mapped_column(Float)  # Vegas over/under

    away_qb: Mapped[Optional["Player"]] = relationship(foreign_keys=[away_qb_id])
    home_qb: Mapped[Optional["Player"]] = relationship(foreign_keys=[home_qb_id])
    # Note: no Team-side relationship for home_team/away_team - Team has two
    # FKs into the same games table (home + away), which needs explicit
    # foreign_keys/overlaps handling on both sides. Skipped for now since
    # nothing currently needs "all games for a team" as an ORM traversal;
    # query Game directly with a where(home_team == x | away_team == x)
    # if/when that's needed.


class SnapCount(Base):
    """One row per player per game, from nflreadpy.load_snap_counts().
    Source keys players by pfr_player_id, not gsis_id - player_id here is
    resolved via players.pfr_id during ingestion and is nullable for the
    rare player nflreadpy can't cross-reference (kept as pfr_player_id
    regardless, so nothing is silently dropped)."""

    __tablename__ = "snap_counts"
    __table_args__ = (
        UniqueConstraint("pfr_player_id", "game_id", name="uq_snap_player_game"),
        Index("idx_snaps_player_season", "player_id", "season"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    player_id: Mapped[Optional[str]] = mapped_column(ForeignKey("players.player_id"))
    pfr_player_id: Mapped[str] = mapped_column(String(16))  # always populated, even if player_id isn't
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"))
    season: Mapped[int] = mapped_column(Integer)
    week: Mapped[int] = mapped_column(Integer)
    game_type: Mapped[str] = mapped_column(String(4))

    player_name: Mapped[str] = mapped_column(String(128))  # from source, in case player_id is null
    position: Mapped[Optional[str]] = mapped_column(String(4))
    team_abbr: Mapped[Optional[str]] = mapped_column(ForeignKey("teams.team_abbr"))
    opponent_abbr: Mapped[Optional[str]] = mapped_column(String(3))

    offense_snaps: Mapped[Optional[float]] = mapped_column(Float)
    offense_pct: Mapped[Optional[float]] = mapped_column(Float)
    defense_snaps: Mapped[Optional[float]] = mapped_column(Float)
    defense_pct: Mapped[Optional[float]] = mapped_column(Float)
    st_snaps: Mapped[Optional[float]] = mapped_column(Float)
    st_pct: Mapped[Optional[float]] = mapped_column(Float)

    player: Mapped[Optional["Player"]] = relationship(back_populates="snap_counts")


class NgsPassing(Base):
    """Next Gen Stats passing, from nflreadpy.load_nextgen_stats(stat_type='passing').
    week=0 rows are nflreadpy's season-to-date aggregate, not a real game -
    filter it out (week > 0) unless you specifically want the season summary."""

    __tablename__ = "ngs_passing"
    __table_args__ = (
        UniqueConstraint("player_id", "season", "week", "season_type", name="uq_ngs_passing"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[str] = mapped_column(ForeignKey("players.player_id"))
    season: Mapped[int] = mapped_column(Integer)
    week: Mapped[int] = mapped_column(Integer)  # 0 = season aggregate row
    season_type: Mapped[str] = mapped_column(String(4))
    team_abbr: Mapped[Optional[str]] = mapped_column(ForeignKey("teams.team_abbr"))

    avg_time_to_throw: Mapped[Optional[float]] = mapped_column(Float)
    avg_completed_air_yards: Mapped[Optional[float]] = mapped_column(Float)
    avg_intended_air_yards: Mapped[Optional[float]] = mapped_column(Float)
    avg_air_yards_differential: Mapped[Optional[float]] = mapped_column(Float)
    aggressiveness: Mapped[Optional[float]] = mapped_column(Float)
    max_completed_air_distance: Mapped[Optional[float]] = mapped_column(Float)
    avg_air_yards_to_sticks: Mapped[Optional[float]] = mapped_column(Float)
    completion_percentage: Mapped[Optional[float]] = mapped_column(Float)
    expected_completion_percentage: Mapped[Optional[float]] = mapped_column(Float)
    completion_percentage_above_expectation: Mapped[Optional[float]] = mapped_column(Float)
    avg_air_distance: Mapped[Optional[float]] = mapped_column(Float)
    max_air_distance: Mapped[Optional[float]] = mapped_column(Float)
    passer_rating: Mapped[Optional[float]] = mapped_column(Float)

    player: Mapped["Player"] = relationship(back_populates="ngs_passing")


class NgsReceiving(Base):
    """Next Gen Stats receiving, from load_nextgen_stats(stat_type='receiving')."""

    __tablename__ = "ngs_receiving"
    __table_args__ = (
        UniqueConstraint("player_id", "season", "week", "season_type", name="uq_ngs_receiving"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[str] = mapped_column(ForeignKey("players.player_id"))
    season: Mapped[int] = mapped_column(Integer)
    week: Mapped[int] = mapped_column(Integer)
    season_type: Mapped[str] = mapped_column(String(4))
    team_abbr: Mapped[Optional[str]] = mapped_column(ForeignKey("teams.team_abbr"))

    avg_cushion: Mapped[Optional[float]] = mapped_column(Float)
    avg_separation: Mapped[Optional[float]] = mapped_column(Float)
    avg_intended_air_yards: Mapped[Optional[float]] = mapped_column(Float)
    percent_share_of_intended_air_yards: Mapped[Optional[float]] = mapped_column(Float)
    catch_percentage: Mapped[Optional[float]] = mapped_column(Float)
    avg_yac: Mapped[Optional[float]] = mapped_column(Float)
    avg_expected_yac: Mapped[Optional[float]] = mapped_column(Float)
    avg_yac_above_expectation: Mapped[Optional[float]] = mapped_column(Float)

    player: Mapped["Player"] = relationship(back_populates="ngs_receiving")


class NgsRushing(Base):
    """Next Gen Stats rushing, from load_nextgen_stats(stat_type='rushing')."""

    __tablename__ = "ngs_rushing"
    __table_args__ = (
        UniqueConstraint("player_id", "season", "week", "season_type", name="uq_ngs_rushing"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[str] = mapped_column(ForeignKey("players.player_id"))
    season: Mapped[int] = mapped_column(Integer)
    week: Mapped[int] = mapped_column(Integer)
    season_type: Mapped[str] = mapped_column(String(4))
    team_abbr: Mapped[Optional[str]] = mapped_column(ForeignKey("teams.team_abbr"))

    efficiency: Mapped[Optional[float]] = mapped_column(Float)
    percent_attempts_gte_eight_defenders: Mapped[Optional[float]] = mapped_column(Float)
    avg_time_to_los: Mapped[Optional[float]] = mapped_column(Float)
    avg_rush_yards: Mapped[Optional[float]] = mapped_column(Float)
    expected_rush_yards: Mapped[Optional[float]] = mapped_column(Float)
    rush_yards_over_expected: Mapped[Optional[float]] = mapped_column(Float)
    rush_yards_over_expected_per_att: Mapped[Optional[float]] = mapped_column(Float)
    rush_pct_over_expected: Mapped[Optional[float]] = mapped_column(Float)

    player: Mapped["Player"] = relationship(back_populates="ngs_rushing")