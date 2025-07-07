import asyncio
import os
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

SEASONS = list(range(2018, 2025))
DATA_DIR = "data"
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")
SCORES_DIR = os.path.join(DATA_DIR, "scores")

async def get_html(url, selector, sleep=5, retries=3):
    html = None
    for i in range(1, retries + 1):
        await asyncio.sleep(sleep * i)
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url)
                print(await page.title())
                html = await page.inner_html(selector)
                await browser.close()
        except PlaywrightTimeout:
            print(f"Timeout error on {url}")
            continue
        else:
            break
    return html

async def scrape_season(season):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = await get_html(url, "#content .filter")
    if not html:
        return

    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all("a")
    hrefs = [l["href"] for l in links if l.has_attr('href')]
    standings_pages = [f"https://www.basketball-reference.com{l}" for l in hrefs]

    for url in standings_pages:
        save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        html = await get_html(url, "#all_schedule")
        if html:
            with open(save_path, "w+", encoding='utf-8') as f:
                f.write(html)

async def scrape_game(standings_file):
    with open(standings_file, "r", encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all("a")
    hrefs = [l.get("href") for l in links]
    box_scores = [l for l in hrefs if l and "boxscore" in l and ".html" in l]
    box_scores = [f"https://www.basketball-reference.com{l}" for l in box_scores]

    for url in box_scores:
        save_path = os.path.join(SCORES_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        html = await get_html(url, "#content")
        if html:
            with open(save_path, "w+", encoding='utf-8') as f:
                f.write(html)

async def main():
    os.makedirs(STANDINGS_DIR, exist_ok=True)
    os.makedirs(SCORES_DIR, exist_ok=True)

    for season in SEASONS:
        await scrape_season(season)

    standings_files = [s for s in os.listdir(STANDINGS_DIR) if s.endswith(".html")]

    for f in standings_files:
        filepath = os.path.join(STANDINGS_DIR, f)
        await scrape_game(filepath)

    print("Scraping completed.")

if __name__ == "__main__":
    asyncio.run(main())
