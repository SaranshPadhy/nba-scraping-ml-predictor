# NBA ML Game Predictor

A machine learning-powered web app to predict NBA game outcomes, featuring both future and past game predictions. The project scrapes, parses, and models NBA data, and provides a Flask-based frontend for visualization.

## Features
- **Data Scraping:** Scrapes NBA game and standings data from Basketball Reference using Playwright and BeautifulSoup.
- **Data Parsing:** Processes raw HTML box scores into structured CSVs for modeling.
- **ML Modeling:** Uses RidgeClassifier and feature selection to predict game outcomes, with rolling window stats for improved accuracy.
- **Frontend:** Flask app displays:
  - Future NBA game predictions (from a mock schedule for now, will be real schedule when next season comes out)
  - Past NBA game predictions, including actual results and whether the prediction was correct

## Project Structure
```
NBA ML Game Predictor/
├── app.py                  # Flask web app
├── get_data.py             # Scrape NBA data
├── parse_data.py           # Parse box scores to CSV
├── predict_past.py         # Train/test on past games, output predictions
├── predict_future.py       # Predict future games from schedule
├── past_nba_games.csv      # Parsed historical games data
├── static/
│   ├── future_predictions.csv
│   └── past_predictions.csv
├── templates/
│   └── index.html          # Frontend template
├── data/                   # Raw scraped data
│   ├── nba_games.csv
│   ├── scores/
│   └── standings/
└── future/
    └── mock_schedule.csv   # Example future schedule
```

## Setup
1. **Clone the repo:**
   ```sh
   git clone <your-repo-url>
   cd NBA-ML-Game-Predictor
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   # And install Playwright browsers:
   playwright install
   ```
3. **Scrape and parse data:**
   ```sh
   python get_data.py
   python parse_data.py
   ```
4. **Generate predictions:**
   ```sh
   python predict_past.py
   python predict_future.py
   ```
5. **Run the web app:**
   ```sh
   python app.py
   # Visit http://127.0.0.1:5000 in your browser
   ```

## Usage
- The homepage displays two tables:
  - **Predicted 2025–2026 NBA Games:** Future predictions
  - **Past NBA Game Predictions:** Past predictions, actual results, and correctness (✔️/❌)

## Notes
- The project uses a mock schedule for future games. Replace `future/mock_schedule.csv` with real schedules when it comes out.
- All data is local; no API keys required.
- Data scraping will take several hours due to basketball-reference's rate-limiting.
- For large datasets, initial data parsing runs may take several minutes.

## Requirements
- Python 3.8+
- pip packages: pandas, scikit-learn, flask, playwright, beautifulsoup4
