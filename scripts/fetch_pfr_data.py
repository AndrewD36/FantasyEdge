import pandas as pd
import os
import time
import random

def fetch_pfr_table(stat_type: str, year: int) -> pd.DataFrame:
    url = f"https://www.pro-football-reference.com/years/{year}/{stat_type}.htm"
    print(f"Fetching {year} {stat_type} stats...")

    try:
        df_list = pd.read_html(url)
        df = df_list[0]
        df = df[df["Player"] != "Player"]
        df = df.fillna(0)
        df["Year"] = year
        df["StatType"] = stat_type

        time.sleep(random.uniform(3.0, 5.0))  #Added delay between requests
        return df

    except Exception as e:
        print(f"Error fetching {stat_type} data for {year}: {e}")
        return pd.DataFrame()

def save_pfr_data(stat_types, years):
    all_data = []
    for stat_type in stat_types:
        for year in years:
            df = fetch_pfr_table(stat_type, year)
            if not df.empty:
                all_data.append(df)

    if all_data:
        full_df = pd.concat(all_data)
        os.makedirs("data/raw", exist_ok=True)
        filename = f"data/raw/pfr_{'_'.join(stat_types)}_{years[0]}_{years[-1]}.csv"
        full_df.to_csv(filename, index=False)
        print(f"\nData saved to {filename}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    print("Select stat types to fetch:")
    print("1: Rushing\n2: Receiving\n3: Passing\n4: All")
    selection = input("Enter numbers separated by commas (e.g., 1,2): ")

    mapping = {"1": "rushing", "2": "receiving", "3": "passing"}
    if "4" in selection:
        stat_types = ["rushing", "receiving", "passing"]
    else:
        stat_types = [mapping[s.strip()] for s in selection.split(",") if s.strip() in mapping]

    start_year = int(input("Enter start year (e.g., 2020): "))
    end_year = int(input("Enter end year (e.g., 2023): "))
    years = list(range(start_year, end_year + 1))

    save_pfr_data(stat_types, years)
    