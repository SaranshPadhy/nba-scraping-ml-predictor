import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib
import shutil

past = pd.read_csv("nba_games.csv", index_col=0)
future = pd.read_csv("future/future_features.csv")

#match feature columns
X = past.drop(columns=["season", "date", "won", "target", "team", "team_opp"], errors="ignore")
X = X.dropna(axis=1)
X = X.loc[:, (X != 0).any(axis=0)] 

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

clf = RidgeClassifier(alpha=1)
clf.fit(X_scaled, past["won"])

#predict future games
future_data = future.drop(columns=["date", "home_team", "away_team"])
future_data = future_data[X.columns]  # same columns as training set
future_scaled = scaler.transform(future_data)

preds = clf.predict(future_scaled)

future["predicted_winner"] = future["home_team"]
future.loc[preds == 0, "predicted_winner"] = future["away_team"]
future.to_csv("future/future_predictions.csv", index=False)
print("Predictions complete.")

shutil.copy("future/future_predictions.csv", "static/future_predictions.csv")
