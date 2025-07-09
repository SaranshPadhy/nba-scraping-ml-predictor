import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/nba_games.csv")
df = df.sort_values("date")
df = df.reset_index(drop = True)

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

        training_data = data[data["season"] < season]
        test_data = data[data["season"] == season]

        model.fit(training_data[predictors], training_data["target"])

        predictions = model.predict(test_data[predictors]) #split training and test data
        predictions = pd.Series(predictions, index=test_data.index)

        combined = pd.concat([test_data["target"], predictions], axis=1)
        combined.columns = ["actual", "prediction"]

        all_predictions.append(combined)
    
    return pd.concat(all_predictions)

def find_team_averages(team):
    rolling = team.rolling(10).mean()
    return rolling

def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name):
    return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

if "date" not in df.columns:
    raise ValueError("Missing 'date' column. Check your CSV.")

df = df.groupby("team", group_keys=False).apply(add_target).reset_index(drop=True)


nulls = pd.isnull(df)
nulls = nulls.sum()
nulls = nulls[nulls > 0]
valid_columns = df.columns[~df.columns.isin(nulls.index)]

df = df[valid_columns].copy()
#cleaned up the null columns for ml

rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)
scaler = MinMaxScaler()

sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction="forward", cv=split)
removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_columns = df.columns[~df.columns.isin(removed_columns)]
df[selected_columns] = scaler.fit_transform(df[selected_columns])
#scale to 1 so ridgeregression perform a bit better

sfs.fit(df[selected_columns], df["target"])
predictor_columns = list(selected_columns[sfs.get_support()])

predictions = backtest(df, rr, predictor_columns)
print(accuracy_score(predictions["actual"], predictions["prediction"]))

df_rolling = df[list(selected_columns) + ["won", "team", "season"]]
df_rolling = df_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages)

rolling_cols = [f"{col}_10" for col in df_rolling.columns]
df_rolling.columns = rolling_cols

df = pd.concat([df, df_rolling], axis=1)
df = df.dropna() #drop null cols for ml alg
df["home_next"] = add_col(df, "home")
df["team_opp_next"] = add_col(df, "team_opp")
df["date_next"] = add_col(df, "date")
df = df.copy() #added columns for the next game for whatever info we know

full = df.merge(df[rolling_cols + ["team_opp_next", "date_next", "team"]], left_on=["team", "date_next"], right_on=["team_opp_next", "date_next"])

removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns #columns i don't want to pass to ml model (esp. target game info)
selected_columns = full.columns[~full.columns.isin(removed_columns)]
sfs.fit(full[selected_columns], full["target"])

predictors = list(selected_columns[sfs.get_support()])

predictions = backtest(full, rr, predictors)
print(accuracy_score(predictions["actual"], predictions["prediction"]))