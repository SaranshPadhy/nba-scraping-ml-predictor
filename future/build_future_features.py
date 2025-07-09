import pandas as pd

# Load past data
df = pd.read_csv("nba_games.csv", index_col=0)
df = df.sort_values("date")

# Rolling stats
def compute_rolling(df, window=10):
    return df.groupby("team", group_keys=False).rolling(window).mean().reset_index(level=0, drop=True)

# Build rolling features for each team
team_stats = compute_rolling(df, window=10)
team_stats["date"] = df["date"]
team_stats["team"] = df["team"]

# Only use latest game per team
latest = team_stats.sort_values("date").groupby("team").tail(1)

# Load schedule
schedule = pd.read_csv("future/mock_schedule.csv")
features = []

for _, row in schedule.iterrows():
    home = row["home_team"]
    away = row["away_team"]
    date = row["date"]

    if home in latest["team"].values and away in latest["team"].values:
        h_stats = latest[latest["team"] == home].drop(columns=["team", "date"]).add_prefix("home_")
        a_stats = latest[latest["team"] == away].drop(columns=["team", "date"]).add_prefix("away_")

        combined = pd.concat([h_stats.reset_index(drop=True), a_stats.reset_index(drop=True)], axis=1)
        combined["date"] = date
        combined["home_team"] = home
        combined["away_team"] = away
        features.append(combined)

future_df = pd.concat(features, ignore_index=True)
future_df.to_csv("future/future_features.csv", index=False)
print("Created future features for prediction.")
