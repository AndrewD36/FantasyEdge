import pandas as pd
import os
import re
import argparse
from pathlib import Path

def clean_numeric_column(series):
    return (
        series.astype(str)
              .str.replace(r'[^\d.\-]', '', regex=True)
              .replace('', '0')
              .astype(float)
    )

def extract_player_and_team(player_str):
    """Extracts player name and team from 'Josh Allen (BUF)'"""
    match = re.match(r"^(.*?)\s*\(([A-Z]{2,3})\)$", str(player_str).strip())
    if match:
        return match.group(1).strip(), match.group(2)
    return str(player_str).strip(), "UNK"

def load_file(path: str):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    elif ext == ".json":
        return pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported input file type: {ext}")

def save_file(df: pd.DataFrame, path: str, fmt: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "json":
        df.to_json(path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")
    print(f"Saved cleaned data to {path}")

def infer_position_from_path(path: str) -> str:
    for pos in ["qb", "rb", "wr", "te", "k", "dst"]:
        if f"/{pos}" in path.lower() or f"\\{pos}" in path.lower() or f"{pos}_" in Path(path).stem.lower():
            return pos
    return "qb"

def clean_fantasypros_stats(input_path: str, output_path: str, output_format: str = "parquet", position: str = None):
    df = load_file(input_path)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Extract player_name and team
    if "player" in df.columns:
        df[['player_name', 'team']] = df['player'].apply(lambda x: pd.Series(extract_player_and_team(x)))
        df = df.drop(columns=['player'])

    # Preserve rostered if it exists
    if 'rostered' not in df.columns and 'rost' in df.columns:
        df.rename(columns={'rost': 'rostered'}, inplace=True)

    # Keep rank for sorting
    if 'rank' in df.columns:
        df['rank'] = clean_numeric_column(df['rank'])
    else:
        df['rank'] = df.groupby(['year', 'week']).cumcount() + 1  # fallback

    # Use fantasy_points_per_game as final fantasy_points
    if 'fantasy_points_per_game' in df.columns:
        df['fantasy_points'] = clean_numeric_column(df['fantasy_points_per_game'])
        df = df.drop(columns=['fantasy_points_per_game'])

    # Drop older aliases if any remain
    for col in ['fpts', 'fpts/g', 'fantasy_points_per_game']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Clean all other numeric columns
    exclude_cols = ['year', 'week', 'player_name', 'team', 'position', 'rank', 'rostered']
    numeric_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype == 'object']
    for col in numeric_cols:
        df[col] = clean_numeric_column(df[col])

    if position:
        df['position'] = position.lower()

    df = df.sort_values(by=['week', 'rank'])

    save_file(df, output_path, output_format)
    return df

def batch_clean(input_dir: str, output_dir: str, output_format: str):
    input_dir = Path(input_dir)
    files = list(input_dir.rglob("*.*"))
    cleaned = 0

    for f in files:
        if f.suffix.lower() not in [".csv", ".json", ".parquet"]:
            continue
        pos = infer_position_from_path(str(f))
        out_path = Path(output_dir) / pos / (f.stem + f".{output_format}")
        clean_fantasypros_stats(str(f), str(out_path), output_format, pos)
        cleaned += 1

    print(f"Batch cleaned {cleaned} files from {input_dir} to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Clean FantasyPros Weekly Stats")

    parser.add_argument("--input", type=str, help="Path to a single file (CSV, JSON, Parquet)")
    parser.add_argument("--output", type=str, help="Path to cleaned file or output directory")
    parser.add_argument("--format", type=str, default="parquet", choices=["csv", "parquet", "json"], help="Output format")
    parser.add_argument("--batch", type=str, help="Directory containing raw files to batch clean")

    args = parser.parse_args()

    if args.batch:
        batch_clean(args.batch, args.output or "cleaned", args.format)
    elif args.input and args.output:
        pos = infer_position_from_path(args.input)
        clean_fantasypros_stats(args.input, args.output, args.format, pos)
    else:
        print("Must specify either --input + --output or --batch + --output")

if __name__ == "__main__":
    main()