import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("nba_games.csv", index_col=0)
df = df.sort_values("date")
df = df.reset_index(drop = True)

del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]

def add_target(team):
    #adds a row: did the team win the next game, needed for ml
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


df = df.groupby("team", group_keys=False).apply(add_target)

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
accuracy_score(predictions["actual"], predictions["prediction"])