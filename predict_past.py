import pandas as pd
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("past_nba_games.csv")
df = df.sort_values("date").reset_index(drop=True)

for col in ["mp.1", "mp_opp.1", "index_opp"]:
    if col in df.columns:
        del df[col]

def add_target(team):
    if "date" in team.columns:
        team = team.sort_values("date")
    team["target"] = team["won"].shift(-1)
    return team

def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []
    seasons = sorted(data["season"].unique())
    for i in range(start, len(seasons), step):
        season = seasons[i]
        train = data[data["season"] < season]
        test = data[data["season"] == season]
        model.fit(train[predictors], train["target"])
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        all_predictions.append(combined)
    return pd.concat(all_predictions)

df = df.groupby("team", group_keys=False).apply(add_target).reset_index(drop=True)
df = df.dropna(subset=["target"]).reset_index(drop=True)
df["target"] = df["target"].astype(int)

base_drop = ["season", "date", "won", "target", "team", "team_opp"]
all_features = [c for c in df.columns if c not in base_drop]

missing_counts = df[all_features].isnull().sum()
complete_features = missing_counts[missing_counts < len(df)].index.tolist()
features = complete_features

df = df.dropna(subset=features).reset_index(drop=True)

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)
sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction="forward", cv=split)
sfs.fit(df[features], df["target"])
preds1 = backtest(df, rr, list(pd.Index(features)[sfs.get_support()]))
print(f"First-stage accuracy: {accuracy_score(preds1['actual'], preds1['prediction'])}")

df_rolling = df.groupby(["team", "season"], group_keys=False)[features].transform(lambda x: x.rolling(10).mean())
df_rolling.columns = [f"{c}_10" for c in df_rolling.columns]
df = pd.concat([df, df_rolling], axis=1)

df = df.dropna(subset=df_rolling.columns).reset_index(drop=True)

df["home_next"] = df.groupby("team", group_keys=False)["home"].shift(-1)
df["team_opp_next"] = df.groupby("team", group_keys=False)["team_opp"].shift(-1)
df["date_next"] = df.groupby("team", group_keys=False)["date"].shift(-1)
df = df.dropna(subset=["home_next", "team_opp_next", "date_next"]).reset_index(drop=True)

opp_roll = df[["team", "date"] + df_rolling.columns.tolist()].copy()
opp_roll = opp_roll.rename(columns={"team": "team_opp_next", "date": "date_next"})
full = df.merge(opp_roll, on=["team_opp_next", "date_next"], how="left", suffixes=("", "_opp"))

drop2 = [c for c in full.columns if full[c].dtype == 'object'] + base_drop + ["home_next", "team_opp_next", "date_next"]
features2 = [c for c in full.columns if c not in drop2]

missing_counts2 = full[features2].isnull().sum()
complete_features2 = missing_counts2[missing_counts2 < len(full)].index.tolist()
features2 = complete_features2

full = full.dropna(subset=features2 + ["target"]).reset_index(drop=True)

sfs2 = SequentialFeatureSelector(rr, n_features_to_select=30, direction="forward", cv=split)
sfs2.fit(full[features2], full["target"])
preds2 = backtest(full, rr, list(pd.Index(features2)[sfs2.get_support()]))

print(f"Second-stage accuracy: {accuracy_score(preds2['actual'], preds2['prediction'])}")


past_pred_df = full.loc[preds2.index, ["date", "home", "team", "team_opp"]].copy()
past_pred_df["predicted_winner"] = full.loc[preds2.index].assign(pred=preds2["prediction"]).apply(
    lambda row: row["team"] if row["pred"] == 1 else row["team_opp"], axis=1)
past_pred_df["actual_winner"] = full.loc[preds2.index].apply(
    lambda row: row["team"] if row["target"] == 1 else row["team_opp"], axis=1)
past_pred_df["home_team"] = past_pred_df.apply(lambda row: row["team"] if row["home"] == 1 else row["team_opp"], axis=1)
past_pred_df["away_team"] = past_pred_df.apply(lambda row: row["team_opp"] if row["home"] == 1 else row["team"], axis=1)
past_pred_df["predicted_correctly"] = past_pred_df["predicted_winner"] == past_pred_df["actual_winner"]
past_pred_df = past_pred_df[["date", "home_team", "away_team", "predicted_winner", "actual_winner", "predicted_correctly"]]
past_pred_df["date"] = pd.to_datetime(past_pred_df["date"]).dt.strftime("%Y-%m-%d")

os.makedirs("static", exist_ok=True)
past_pred_df.to_csv("static/past_predictions.csv", index=False)
print("Past predictions saved to static/past_predictions.csv")
