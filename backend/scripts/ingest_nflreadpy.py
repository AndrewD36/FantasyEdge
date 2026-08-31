from __future__ import annotations
 
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import nflreadpy as nfl
from sqlalchemy import create_engine
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import Base, Team, Player, PlayerTeamSeason, PlayerStat, Game, SnapCount, NgsPassing, NgsReceiving, NgsRushing
from app.schemas import (
    TeamSchema, PlayerCreate, PlayerStatCreate,
    SnapCountCreate, NgsPassingCreate, NgsReceivingCreate, NgsRushingCreate,
)

DB_URL = "sqlite:///db/fantasyedge.db"

def upsert(session: Session, model, rows: list[dict[str, Any]], conflict_cols: list[str]) -> None:
    """Batched upsert. SQLite caps bound parameters per statement (~32k
    by default, sometimes lower), so a single INSERT with thousands of
    wide rows can blow past that limit. Chunk rows so each statement
    stays well under the cap regardless of column count."""
    if not rows:
        return
    table = model.__table__
    n_cols = len(rows[0])
    # keep comfortably under SQLite's limit even for wide tables like player_stats
    batch_size = max(1, 900 // max(n_cols, 1))
    update_cols = {c.name for c in table.columns if c.name not in conflict_cols}
 
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        stmt = sqlite_insert(table).values(batch)
        if update_cols:
            stmt = stmt.on_conflict_do_update(
                index_elements=conflict_cols,
                set_={name: stmt.excluded[name] for name in update_cols},
            )
        else:
            # every column is part of the conflict key (pure bridge table) -
            # nothing to update, just skip duplicates
            stmt = stmt.on_conflict_do_nothing(index_elements=conflict_cols)
        session.execute(stmt)

def load_teams(session: Session) -> None:
    df = nfl.load_teams()
    rows = []
    for r in df.iter_rows(named=True):
        team = TeamSchema(
            team_abbr=r["team_abbr"],
            team_name=r["team_name"],
            conference=r["team_conf"],
            division=r["team_division"],
        )
        rows.append(team.model_dump())
    upsert(session, Team, rows, conflict_cols=["team_abbr"])
    print(f"  teams: {len(rows)} upserted")

def load_players(session: Session, seasons: list[int]) -> None:
    players_df = nfl.load_players()
    rosters_df = nfl.load_rosters(seasons)
 
    # rosters has one row per player per team-stint; take the most recent
    # non-null sleeper_id/yahoo_id per gsis_id
    id_map = (
        rosters_df.select(["gsis_id", "sleeper_id", "yahoo_id"])
        .filter(pl.col("gsis_id").is_not_null())
        .unique(subset=["gsis_id"], keep="last")
    )
    sleeper_by_gsis = dict(
        zip(id_map["gsis_id"].to_list(), id_map["sleeper_id"].to_list())
    )
    yahoo_by_gsis = dict(
        zip(id_map["gsis_id"].to_list(), id_map["yahoo_id"].to_list())
    )
 
    rows, skipped = [], 0
    for r in players_df.iter_rows(named=True):
        birth_date = None
        if r["birth_date"]:
            try:
                birth_date = datetime.strptime(r["birth_date"], "%Y-%m-%d").date()
            except (ValueError, TypeError):
                pass
 
        try:
            player = PlayerCreate(
                player_id=r["gsis_id"],
                full_name=r["display_name"],
                first_name=r["first_name"],
                last_name=r["last_name"],
                position=r["position"],
                birth_date=birth_date,
                height=r["height"],
                weight=r["weight"],
                college=r["college_name"],
                draft_year=r["draft_year"],
                draft_round=r["draft_round"],
                draft_pick=r["draft_pick"],
                rookie_season=r["rookie_season"],
                status=r["status"],
                espn_id=r["espn_id"],
                sleeper_id=sleeper_by_gsis.get(r["gsis_id"]),
                yahoo_id=yahoo_by_gsis.get(r["gsis_id"]),
            )
        except Exception as e:
            skipped += 1
            continue
        rows.append(player.model_dump())
 
    upsert(session, Player, rows, conflict_cols=["player_id"])
    print(f"  players: {len(rows)} upserted, {skipped} skipped (validation failures)")

def load_player_team_seasons(session: Session, seasons: list[int]) -> None:
    df = nfl.load_rosters_weekly(seasons)
    rows = []
    seen = set()
    for r in df.iter_rows(named=True):
        if not r["gsis_id"] or not r["team"]:
            continue
        key = (r["gsis_id"], r["season"], r["team"], r["week"] or 0)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "player_id": r["gsis_id"],
                "season": r["season"],
                "team_abbr": r["team"],
                "week": r["week"] or 0,
            }
        )
    upsert(
        session,
        PlayerTeamSeason,
        rows,
        conflict_cols=["player_id", "season", "team_abbr", "week"],
    )
    print(f"  player_team_seasons: {len(rows)} upserted")

def load_player_stats(session: Session, seasons: list[int]) -> None:
    df = nfl.load_player_stats(seasons, summary_level="week")
 
    # only keep rows whose player_id actually exists in players table,
    # otherwise the FK insert fails
    known_ids = {row[0] for row in session.query(Player.player_id).all()}
 
    rows, skipped = [], 0
    for r in df.iter_rows(named=True):
        if r["player_id"] not in known_ids:
            skipped += 1
            continue
        try:
            stat = PlayerStatCreate(
                player_id=r["player_id"],
                season=r["season"],
                week=r["week"],
                season_type=r["season_type"],
                team_abbr=r["team"],
                opponent_abbr=r["opponent_team"],
                completions=r["completions"] or 0,
                attempts=r["attempts"] or 0,
                passing_yards=r["passing_yards"] or 0,
                passing_tds=r["passing_tds"] or 0,
                interceptions=r["passing_interceptions"] or 0,
                sacks=r["sacks_suffered"] or 0,
                sack_yards=r["sack_yards_lost"] or 0,
                carries=r["carries"] or 0,
                rushing_yards=r["rushing_yards"] or 0,
                rushing_tds=r["rushing_tds"] or 0,
                rushing_fumbles=r["rushing_fumbles"] or 0,
                targets=r["targets"] or 0,
                receptions=r["receptions"] or 0,
                receiving_yards=r["receiving_yards"] or 0,
                receiving_tds=r["receiving_tds"] or 0,
                receiving_fumbles=r["receiving_fumbles"] or 0,
                fantasy_points=r["fantasy_points"],
                fantasy_points_ppr=r["fantasy_points_ppr"],
            )
        except Exception:
            skipped += 1
            continue
        rows.append(stat.model_dump())
 
    upsert(
        session,
        PlayerStat,
        rows,
        conflict_cols=["player_id", "season", "week", "season_type"],
    )
    print(f"  player_stats: {len(rows)} upserted, {skipped} skipped")

def load_games(session: Session, seasons: list[int]) -> None:
    df = nfl.load_schedules(seasons)
    rows = []
    for r in df.iter_rows(named=True):
        rows.append(
            {
                "game_id": r["game_id"],
                "season": r["season"],
                "week": r["week"],
                "game_type": r["game_type"],
                "gameday": _parse_date(r["gameday"]),
                "weekday": r["weekday"],
                "gametime": r["gametime"],
                "away_team": r["away_team"],
                "home_team": r["home_team"],
                "away_score": r["away_score"],
                "home_score": r["home_score"],
                "result": r["result"],
                "total": r["total"],
                "overtime": bool(r["overtime"]),
                "div_game": bool(r["div_game"]),
                "roof": r["roof"],
                "surface": r["surface"],
                "temp": r["temp"],
                "wind": r["wind"],
                "away_qb_id": r["away_qb_id"],
                "home_qb_id": r["home_qb_id"],
                "away_qb_name": r["away_qb_name"],
                "home_qb_name": r["home_qb_name"],
                "away_coach": r["away_coach"],
                "home_coach": r["home_coach"],
                "referee": r["referee"],
                "stadium": r["stadium"],
                "spread_line": r["spread_line"],
                "total_line": r["total_line"],
            }
        )
    upsert(session, Game, rows, conflict_cols=["game_id"])
    print(f"  games: {len(rows)} upserted")

def load_snap_counts(session: Session, seasons: list[int]) -> None:
    df = nfl.load_snap_counts(seasons)
    pfr_to_gsis = _pfr_id_map()
 
    rows = []
    for r in df.iter_rows(named=True):
        try:
            snap = SnapCountCreate(
                player_id=pfr_to_gsis.get(r["pfr_player_id"]),
                pfr_player_id=r["pfr_player_id"],
                game_id=r["game_id"],
                season=r["season"],
                week=r["week"],
                game_type=r["game_type"],
                player_name=r["player"],
                position=r["position"],
                team_abbr=r["team"],
                opponent_abbr=r["opponent"],
                offense_snaps=r["offense_snaps"],
                offense_pct=r["offense_pct"],
                defense_snaps=r["defense_snaps"],
                defense_pct=r["defense_pct"],
                st_snaps=r["st_snaps"],
                st_pct=r["st_pct"],
            )
        except Exception:
            continue
        rows.append(snap.model_dump())
 
    upsert(session, SnapCount, rows, conflict_cols=["pfr_player_id", "game_id"])
    print(f"  snap_counts: {len(rows)} upserted")
 
 
def _pfr_id_map() -> dict[str, str]:
    """players.pfr_id isn't a mapped column on Player (we only kept a subset
    of load_players()'s fields) - so build the pfr_id -> gsis_id lookup
    straight from nflreadpy rather than the DB."""
    players_df = nfl.load_players()
    players_df = players_df.filter(players_df["pfr_id"].is_not_null())
    return dict(zip(players_df["pfr_id"].to_list(), players_df["gsis_id"].to_list()))
 
 
def _parse_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None

_NGS_CONFIG = {
    "passing": (NgsPassing, NgsPassingCreate, [
        "avg_time_to_throw", "avg_completed_air_yards", "avg_intended_air_yards",
        "avg_air_yards_differential", "aggressiveness", "max_completed_air_distance",
        "avg_air_yards_to_sticks", "completion_percentage", "expected_completion_percentage",
        "completion_percentage_above_expectation", "avg_air_distance", "max_air_distance",
        "passer_rating",
    ]),
    "receiving": (NgsReceiving, NgsReceivingCreate, [
        "avg_cushion", "avg_separation", "avg_intended_air_yards",
        "percent_share_of_intended_air_yards", "catch_percentage", "avg_yac",
        "avg_expected_yac", "avg_yac_above_expectation",
    ]),
    "rushing": (NgsRushing, NgsRushingCreate, [
        "efficiency", "percent_attempts_gte_eight_defenders", "avg_time_to_los",
        "avg_rush_yards", "expected_rush_yards", "rush_yards_over_expected",
        "rush_yards_over_expected_per_att", "rush_pct_over_expected",
    ]),
}
 
 
def load_nextgen_stats(session: Session, seasons: list[int]) -> None:
    known_ids = {row[0] for row in session.query(Player.player_id).all()}
 
    for stat_type, (model, schema, stat_fields) in _NGS_CONFIG.items():
        df = nfl.load_nextgen_stats(seasons, stat_type=stat_type)
        rows, skipped = [], 0
        for r in df.iter_rows(named=True):
            if not r["player_gsis_id"] or r["player_gsis_id"] not in known_ids:
                skipped += 1
                continue
            try:
                entry = schema(
                    player_id=r["player_gsis_id"],
                    season=r["season"],
                    week=r["week"],
                    season_type=r["season_type"],
                    team_abbr=r["team_abbr"],
                    **{f: r[f] for f in stat_fields},
                )
            except Exception:
                skipped += 1
                continue
            rows.append(entry.model_dump())
 
        upsert(session, model, rows, conflict_cols=["player_id", "season", "week", "season_type"])
        print(f"  ngs_{stat_type}: {len(rows)} upserted, {skipped} skipped")

def main(seasons: list[int]) -> None:
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)
 
    with Session(engine) as session:
        print("Loading teams...")
        load_teams(session)
        session.commit()
 
        print("Loading players...")
        load_players(session, seasons)
        session.commit()
 
        print("Loading player-team-season bridge...")
        load_player_team_seasons(session, seasons)
        session.commit()
 
        print("Loading player stats...")
        load_player_stats(session, seasons)
        session.commit()
 
        print("Loading games (schedules)...")
        load_games(session, seasons)
        session.commit()
 
        print("Loading snap counts...")
        load_snap_counts(session, seasons)
        session.commit()
 
        print("Loading Next Gen Stats...")
        load_nextgen_stats(session, seasons)
        session.commit()
 
    print("Done.")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    args = parser.parse_args()
    main(args.seasons)