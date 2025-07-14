import argparse
import requests as r
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random as ran
from datetime import datetime as dt

BASE_URL = "https://www.fantasypros.com/nfl/stats/{position}.php"
HEADERS = {"User-Agent": "Mozilla/5.0"}
ALL_POSITIONS = ["qb", "rb", "wr", "te", "k", "dst"]

def fetch_week_data(position: str, week: int, year: int) -> pd.DataFrame:
    url = f"{BASE_URL}?range=week&week={week}&year={year}".format(position=position)
    print(f"Fetching {position.upper()} stats: Week {week}, Year {year}")
    response = r.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"HTTP {response.status_code} for {url}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find("table", {"id": "data"})
    if not table:
        print(f"No table found for {position} week {week}, year {year}")
        return pd.DataFrame()

    raw_headers = [th.get_text(strip=True) for th in table.find_all("th")]
    headers = prefix_headers(raw_headers, position)
    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not cells:
            continue
        if len(cells) < len(headers):
            # pad missing cells
            cells += [""] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            # trim extra cells
            cells = cells[:len(headers)]
        rows.append(cells)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=headers)
    df.insert(0, "week", week)
    df.insert(0, "year", year)
    return df

def prefix_headers(headers, position):
    """Adds stat categories for clarity and deduplication."""
    if position == "qb":
        return (
            ["Rank", "Player"] +
            ["Pass_" + h for h in headers[2:8]] +     # CMP to INT
            ["INT", "Sacks"] +
            ["Rush_ATT", "Rush_YDS", "Rush_TD"] +
            ["Fumbles", "Games", "Fantasy_Points", "Fantasy_Points_Per_Game", "Rostered"]
        )
    elif position == "rb":
        return (
            ["Rank", "Player"] +
            ["Rush_ATT", "Rush_YDS", "Rush_TD"] +
            ["Rec_TGT", "Rec_REC", "Rec_YDS", "Rec_TD"] +
            ["FL", "G", "FPTS", "FPTS/G", "ROST"]
        )
    elif position in ["wr", "te"]:
        return (

        )
    elif position == "k":
        return headers  # Usually no duplicate names
    elif position == "dst":
        return headers  # Usually unique column names
    else:
        return headers

def scrape(position: str, year: int, weeks: list[int], output_dir: str, file_format: str = "csv"):
    all_weeks = []
    print(f"\nScraping {position.upper()} for {year}...")

    for week in weeks:
        try:
            df = fetch_week_data(position, week, year)
            if not df.empty:
                all_weeks.append(df)
            time.sleep(ran.uniform(0.5, 1.5))
        except Exception as e:
            print(f"Error: {e}")

    if all_weeks:
        year_df = pd.concat(all_weeks, ignore_index=True)
        out_dir = os.path.join(output_dir, position)
        os.makedirs(out_dir, exist_ok=True)
        ext = file_format.lower()
        output_path = os.path.join(out_dir, f"{position}_{year}.{ext}")

        if ext == "csv":
            year_df.to_csv(output_path, index=False)
        elif ext == "parquet":
            year_df.to_parquet(output_path, index=False)
        elif ext == "json":
            year_df.to_json(output_path, orient="records", lines=True)

        print(f"Saved {len(year_df)} rows → {output_path}")
    else:
        print(f"No data scraped for {position.upper()} in {year}")


def parse_range_or_all(value: str, full_range: list[int]) -> list[int]:
    if value == "all":
        return full_range
    elif "-" in value:
        start, end = map(int, value.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(value)]


def main():
    parser = argparse.ArgumentParser(description="FantasyPros NFL Weekly Stats Scraper")

    parser.add_argument("--position", type=str, default="all",
                        help="qb, rb, wr, te, k, dst or 'all'")
    parser.add_argument("--year", type=str, default="all",
                        help="Single year (e.g. 2023), range (e.g. 2020-2024), or 'all'")
    parser.add_argument("--week", type=str, default="all",
                        help="Single week (e.g. 2), or 'all' (1-18)")   
    parser.add_argument("--output", type=str, default="data",
                    help="Directory to store scraped files (default: 'data')")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "parquet", "json"],
                    help="Output format: csv, parquet, or json (default: csv)")

    args = parser.parse_args()

    # Parse positions
    positions = ALL_POSITIONS if args.position == "all" else [args.position.lower()]
    for pos in positions:
        if pos not in ALL_POSITIONS:
            raise ValueError(f"Invalid position: {pos}")

    # Parse years
    full_years = list(range(2002, dt.now().year-1))  # Can be extended
    years = parse_range_or_all(args.year, full_years)

    # Parse weeks
    weeks = parse_range_or_all(args.week, list(range(1, 18)))

    # Scrape
    for pos in positions:
        for year in years:
            scrape(pos, year, weeks, output_dir=args.output, file_format=args.format)


if __name__ == "__main__":
    main()