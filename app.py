from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__)

@app.route("/")
def index():
    future_csv = os.path.join("static", "future_predictions.csv")
    past_csv = os.path.join("static", "past_predictions.csv")

    if os.path.exists(future_csv):
        df_future = pd.read_csv(future_csv)
        future_predictions = df_future.to_dict(orient="records")
    else:
        future_predictions = []

    if os.path.exists(past_csv):
        df_past = pd.read_csv(past_csv)
        past_predictions = df_past.to_dict(orient="records")
    else:
        past_predictions = []

    return render_template("index.html", games=future_predictions, past_games=past_predictions)

if __name__ == "__main__":
    app.run(debug=True)
