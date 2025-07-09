import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os

past = pd.read_csv("data/nba_games.csv")
past["date"] = pd.to_datetime(past["date"])
past = past.sort_values("date")
past = past.dropna()

schedule = pd.read_csv("future/mock_schedule.csv")

def compute_rolling(df, window=3):
    numeric = df.select_dtypes(include="number")
    rolled = df.groupby("team", group_keys=False)[numeric.columns].rolling(window).mean()
    return rolled.reset_index(level=0, drop=True)

team_stats = compute_rolling(past, window=10)
team_stats["date"] = past["date"]
team_stats["team"] = past["team"]

latest = team_stats.sort_values("date").groupby("team").tail(1)

train_rows = []
seen_games = set()

for idx, row in past.iterrows():
    key = tuple(sorted([row["team"], row["team_opp"], str(row["date"])]))
    if key in seen_games:
        continue
    seen_games.add(key)

    team = row["team"]
    opp = row["team_opp"]
    date = row["date"]
    home = row["home"]
    won = row["won"]

    team_row = team_stats[(team_stats["team"] == team) & (team_stats["date"] == date)]
    opp_row = team_stats[(team_stats["team"] == opp) & (team_stats["date"] == date)]

    if team_row.empty or opp_row.empty:
        continue

    team_row = team_row.drop(columns=["team", "date"]).add_prefix("home_" if home == 1 else "away_")
    opp_row = opp_row.drop(columns=["team", "date"]).add_prefix("away_" if home == 1 else "home_")

    merged = pd.concat([team_row.reset_index(drop=True), opp_row.reset_index(drop=True)], axis=1)
    merged["target"] = won
    train_rows.append(merged)

train_df = pd.concat(train_rows, ignore_index=True).dropna()

X = train_df.drop(columns=["target"])
y = train_df["target"]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

rr = RidgeClassifier(alpha=1)
rr.fit(X, y)

future_rows = []

for _, row in schedule.iterrows():
    home = row["home_team"]
    away = row["away_team"]
    date = row["date"]

    if home not in latest["team"].values or away not in latest["team"].values:
        continue

    h_stats = latest[latest["team"] == home].drop(columns=["team", "date"]).add_prefix("home_")
    a_stats = latest[latest["team"] == away].drop(columns=["team", "date"]).add_prefix("away_")

    combined = pd.concat([h_stats.reset_index(drop=True), a_stats.reset_index(drop=True)], axis=1)
    combined["date"] = date
    combined["home_team"] = home
    combined["away_team"] = away
    future_rows.append(combined)

future_df = pd.concat(future_rows, ignore_index=True).dropna()

feature_cols = train_df.drop(columns=["target"]).columns
future_scaled = scaler.transform(future_df[feature_cols])
preds = rr.predict(future_scaled)

future_df["predicted_winner"] = future_df["home_team"]
future_df.loc[preds == 0, "predicted_winner"] = future_df["away_team"]

output = future_df[["date", "home_team", "away_team", "predicted_winner"]]
os.makedirs("static", exist_ok=True)
output.to_csv("static/future_predictions.csv", index=False)
print("Predictions complete.")
