import argparse
from datetime import datetime
from pathlib import Path

import nflreadpy as nfl

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "db"
DB_PATH = DB_DIR / "nfl.db"

DATABASE_URL = f"sqlite:///{DB_PATH}"

TABLE_CONFIG = {
    "play_by_play": {
        "loader": nfl.load_pbp,
        "uses_seasons": True,
        "uses_summary_level": False,
        "uses_stat_type": False,
    },
    "player_stats": {
        "loader": nfl.load_player_stats,
        "uses_seasons": True,
        "uses_summary_level": True,
        "uses_stat_type": False,
    },
    "team_stats": {
        "loader": nfl.load_team_stats,
        "uses_seasons": True,
        "uses_summary_level": True,
        "uses_stat_type": False,
    },
    "schedules": {
        "loader": nfl.load_player_stats,
        "uses_seasons": True,
        "uses_summary_level": True,
        "uses_stat_type": False,
    },
    "players": {
        "loader": nfl.load_players,
        "uses_seasons": False,
        "uses_summary_level": False,
        "uses_stat_type": False,
    },
    "rosters": {
        "loader": nfl.load_rosters,
        "uses_seasons": True,
        "uses_summary_level": False,
        "uses_stat_type": False,
    },
    "snap_counts": {
        "loader": nfl.load_snap_counts,
        "uses_seasons": True,
        "uses_summary_level": False,
        "uses_stat_type": False,
    },
    "nextgen_stats": {
        "loader": nfl.load_nextgen_stats,
        "uses_seasons": True,
        "uses_summary_level": False,
        "uses_stat_type": True,
    },
    "participation": {
        "loader": nfl.load_participation,
        "uses_seasons": True,
        "uses_summary_level": False,
        "uses_stat_type": False,
    },
    "injuries": {
        "loader": nfl.load_injuries,
        "uses_seasons": True,
        "uses_summary_level": False,
        "uses_stat_type": False,
    },
    "officials": {
        "loader": nfl.load_officials,
        "uses_seasons": True,
        "uses_summary_level": False,
        "uses_stat_type": False,
    },
}

def load_table(
    table_name: str,
    seasons: list[int],
    summary_level: str,
    stat_type: str
):
    config = TABLE_CONFIG[table_name]

    loader = config["loader"]

    kwargs = {}

    if config["uses_seasons"]:
        kwargs["seasons"] = seasons

    if config["uses_summary_level"]:
        kwargs["summary_level"] = summary_level

    if config["uses_stat_type"]:
        kwargs["stat_type"] = stat_type

    print(f"Downloading {table_name} data...")

    data = loader(**kwargs)

    print(f"Downloaded {data.height:,} rows")

    data.write_database(
        table_name=table_name,
        connection=DATABASE_URL,
        if_table_exists="replace",
    )

    print(f"Saved {table_name}\n")

def parse_args():
    current_year = datetime.now().year

    parser = argparse.ArgumentParser(
        description="Load nflreadpy datasets into SQLite."
    )

    parser.add_argument(
        "--tables",
        nargs="+",
        choices=TABLE_CONFIG.keys(),
        default=list(TABLE_CONFIG.keys()),
        help="Tables to load (). Defaults to all tables.",
    )

    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(range(2020, current_year + 1)),
        help=f"Seasons to load. Defaults to 2020-{current_year}.",
    )

    parser.add_argument(
        "--summary_level",
        choices=["week", "reg", "post", "reg+post"],
        default="week",
        help="Summary level for datasets that support it (week, reg, post, season). Defaults to season.",
    )

    parser.add_argument(
        "--stat_type",
        choices=["passing", "receiving", "rushing"],
        default="passing",
        help="Stat type for datasets that support it (passing, receiving, rushing). Defaults to passing.",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    DB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Database: {DB_PATH}")
    print(f"Tables: {', '.join(args.tables)}")
    print(f"Seasons: {args.seasons}")
    print(f"Summary Level: {args.summary_level}")
    print(f"Stat Type: {args.stat_type}\n")

    for table_name in args.tables:
        load_table(
            table_name=table_name,
            seasons=args.seasons,
            summary_level=args.summary_level,
            stat_type=args.stat_type,
        )

    print(f"SQLite database updated: {DB_PATH}")

if __name__ == "__main__":
    main()