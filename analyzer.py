import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the FinBERT model and tokenizer from Hugging Face
# FinBERT is specifically pre-trained on financial communication
TOKENIZER_NAME = "ProsusAI/finbert"
MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def fetch_financial_news(ticker: str) -> list:
    """
    Fetches the latest financial news headlines for a ticker using Yahoo Finance RSS Feed.
    No API keys required!
    """
    url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    headlines = []
    
    try:
        # Request the RSS feed
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
            
        # Parse XML
        root = ET.fromstring(xml_data)
        for item in root.findall('.//item'):
            title = item.find('title').text
            pub_date = item.find('pubDate').text
            link = item.find('link').text
            
            headlines.append({
                "Headline": title,
                "Published": pub_date,
                "Link": link
            })
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        
    return headlines[:10]  # Limit to the 10 most recent news articles

def analyze_sentiment(headlines_list: list) -> pd.DataFrame:
    """
    Takes a list of headlines, runs them through FinBERT, and returns a Pandas DataFrame 
    with positive, negative, and neutral probability scores.
    """
    if not headlines_list:
        return pd.DataFrame()

    df = pd.DataFrame(headlines_list)
    headlines = df['Headline'].tolist()
    
    # Tokenize input text
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
    
    # Run predictions without calculating gradients (faster)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # FinBERT labels: 0 -> Positive, 1 -> Negative, 2 -> Neutral
    df['Positive %'] = predictions[:, 0].numpy() * 100
    df['Negative %'] = predictions[:, 1].numpy() * 100
    df['Neutral %'] = predictions[:, 2].numpy() * 100
    
    # Assign the dominant sentiment label
    labels = ['Positive', 'Negative', 'Neutral']
    dominant_idx = torch.argmax(predictions, dim=-1).numpy()
    df['Sentiment'] = [labels[idx] for idx in dominant_idx]
    
    return df