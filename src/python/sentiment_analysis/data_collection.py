"""
Data Collection for Sentiment Analysis

Purpose:
This module collects textual data from financial news, earnings call transcripts, and social media
platforms (e.g., Twitter, Reddit). The collected data is used for sentiment analysis to gauge public
sentiment about specific stocks or the market as a whole.

Role in FORESIGHT:
- Provides raw textual data for sentiment analysis.
- Integrates with external APIs and web scraping tools to fetch real-time data.
- Supports the broader sentiment analysis pipeline.

Key Features:
- Web scraping for financial news and social media data.
- API integration for earnings call transcripts.
- Data storage for further processing.
"""

import requests
from bs4 import BeautifulSoup

def fetch_financial_news(url: str) -> str:
    """
    Fetch financial news articles from a given URL.

    Args:
        url (str): URL of the financial news website.

    Returns:
        str: Text content of the news articles.
    """
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("article")
        return " ".join([article.get_text() for article in articles])
    else:
        raise Exception(f"Failed to fetch financial news: {response.status_code}")

def fetch_twitter_data(query: str, api_key: str) -> list:
    """
    Fetch tweets related to a specific query using the Twitter API.

    Args:
        query (str): Search query (e.g., stock ticker or company name).
        api_key (str): API key for the Twitter API.

    Returns:
        list: List of tweets as strings.
    """
    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return [tweet["text"] for tweet in response.json()["data"]]
    else:
        raise Exception(f"Failed to fetch Twitter data: {response.status_code}")

# Example usage
if __name__ == "__main__":
    news_url = "https://www.example-financial-news.com"
    news_text = fetch_financial_news(news_url)
    print("Financial news text:", news_text[:500])  # Print first 500 characters

    twitter_api_key = "your_twitter_api_key"
    tweets = fetch_twitter_data("AAPL", twitter_api_key)
    print("Tweets about AAPL:", tweets[:5])  # Print first 5 tweets