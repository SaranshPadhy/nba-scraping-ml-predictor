from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__)

@app.route("/")
def index():
    csv_path = os.path.join("static", "future_predictions.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        predictions = df.to_dict(orient="records")
    else:
        predictions = []

    return render_template("index.html", games=predictions)

if __name__ == "__main__":
    app.run(debug=True)
