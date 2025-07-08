import os
import pandas as pd
from bs4 import BeautifulSoup

SCORE_DIR = "data/scores"

def parse_html(box_score_path):
    with open(box_score_path, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    [s.decompose() for s in soup.select("tr.over_header")]
    [s.decompose() for s in soup.select("tr.thead")]
    return soup

def read_line_score(soup):
    try:
        line_score = pd.read_html(str(soup), attrs={"id": "line_score"})[0]
    except ValueError:
        return None
    cols = list(line_score.columns)
    cols[0] = "team"
    cols[-1] = "total"
    line_score.columns = cols
    return line_score[["team", "total"]]

def read_stats(soup, team, stat):
    try:
        df = pd.read_html(str(soup), attrs={"id": f"box-{team}-game-{stat}"}, index_col=0)[0]
    except ValueError:
        return pd.DataFrame()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def read_season_info(soup):
    nav = soup.select_one("#bottom_nav_container")
    if not nav:
        return None
    hrefs = [a["href"] for a in nav.find_all("a") if "href" in a.attrs]
    if len(hrefs) < 2:
        return None
    season = os.path.basename(hrefs[1]).split("_")[0]
    return season

def process_box_scores():
    box_scores = [os.path.join(SCORE_DIR, f) for f in os.listdir(SCORE_DIR) if f.endswith(".html")]
    base_cols = None
    games = []

    for idx, box_score in enumerate(box_scores, 1):
        soup = parse_html(box_score)
        line_score = read_line_score(soup)
        if line_score is None or line_score.empty:
            continue
        teams = list(line_score["team"])

        summaries = []
        for team in teams:
            basic = read_stats(soup, team, "basic")
            adv = read_stats(soup, team, "advanced")

            if basic.empty or adv.empty:
                continue

            totals = pd.concat([basic.iloc[-1, :], adv.iloc[-1, :]])
            totals.index = totals.index.str.lower()

            maxes = pd.concat([basic.iloc[:-1, :].max(), adv.iloc[:-1, :].max()])
            maxes.index = maxes.index.str.lower() + "_max"
            summary = pd.concat([totals, maxes])

            if base_cols is None:
                base_cols = list(summary.index.drop_duplicates(keep="first"))
                base_cols = [b for b in base_cols if "bpm" not in b]

            summary = summary[base_cols]
            summaries.append(summary)

        if len(summaries) != 2:
            continue

        summary = pd.concat(summaries, axis=1).T
        game = pd.concat([summary, line_score], axis=1)
        game["home"] = [0, 1]
        game_opponent = game.iloc[::-1].reset_index(drop=True)
        game_opponent.columns = [f"{col}_opp" for col in game_opponent.columns]

        full_game = pd.concat([game.reset_index(drop=True), game_opponent], axis=1)
        season = read_season_info(soup)
        if season is None:
            continue
        full_game["season"] = season
        full_game["date"] = os.path.basename(box_score)[:8]
        full_game["date"] = pd.to_datetime(full_game["date"], format="%Y%m%d")
        full_game["won"] = full_game["total"] > full_game["total_opp"]
        games.append(full_game)

        if idx % 100 == 0:
            print(f"{idx} / {len(box_scores)} processed.")

    if games:
        games_df = pd.concat(games, ignore_index=True)
        return games_df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    games_df = process_box_scores()
    games_df.to_csv("nba_games.csv")
    if not games_df.empty:
        print("Data processing complete. DataFrame shape:", games_df.shape)
    else:
        print("No data processed.")
