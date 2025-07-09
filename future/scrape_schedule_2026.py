import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://www.basketball-reference.com/leagues/NBA_2026_games.html"

def scrape_schedule():
    response = requests.get(URL)
    soup = BeautifulSoup(response.text, 'html.parser')

    schedule = []
    for link in soup.select("#content a"):
        if "NBA_2026_games" in link.get("href", "") and "html" in link["href"]:
            schedule.append("https://www.basketball-reference.com" + link["href"])

    games = []
    for url in schedule:
        res = requests.get(url)
        month_soup = BeautifulSoup(res.text, "html.parser")
        table = month_soup.find("table")
        if table:
            df = pd.read_html(str(table))[0]
            for _, row in df.iterrows():
                if row.get("Home/Neutral") and row.get("Visitor/Neutral"):
                    games.append({
                        "date": pd.to_datetime(row["Date"]),
                        "home_team": row["Home/Neutral"],
                        "away_team": row["Visitor/Neutral"]
                    })

    pd.DataFrame(games).to_csv("future/schedule_2026.csv", index=False)
    print("Saved future schedule")

if __name__ == "__main__":
    scrape_schedule()
