import streamlit as st
import pandas as pd
from analyzer import fetch_financial_news, analyze_sentiment

# Set page layout and title
st.set_page_config(page_title="Deutsche Bank Risk Analyst Tool", page_icon="📈", layout="wide")

st.title("📈 AI-Powered Financial News Sentiment & Risk Analyzer")
st.markdown("""
This tool helps risk analysts scan real-time financial news and automatically flag potential market/credit risks using **FinBERT** (Financial Bidirectional Encoder Representations from Transformers).
""")

# Sidebar for inputs
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., DB, AAPL, MSFT, TSLA):", value="DB").upper().strip()

analyze_btn = st.sidebar.button("Run Risk Assessment")

if analyze_btn:
    if not ticker:
        st.warning("Please enter a valid ticker symbol.")
    else:
        with st.spinner(f"Fetching latest news and running NLP analysis for {ticker}..."):
            # 1. Fetch news
            raw_news = fetch_financial_news(ticker)
            
            if not raw_news:
                st.error(f"No news found or could not fetch news for {ticker}. Ensure the ticker is correct.")
            else:
                # 2. Analyze Sentiment
                results_df = analyze_sentiment(raw_news)
                
                # Calculate Risk Metrics
                avg_neg = results_df['Negative %'].mean()
                avg_pos = results_df['Positive %'].mean()
                
                # Visual Metric Cards
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Overall Risk Status", 
                              value="HIGH RISK 🚨" if avg_neg > 40 else "STABLE ✅",
                              delta=f"Avg Negative Sentiment: {avg_neg:.1f}%", 
                              delta_color="inverse")
                with col2:
                    st.metric(label="Average Positive Sentiment Score", value=f"{avg_pos:.1f}%")
                with col3:
                    st.metric(label="Total Analyzed Articles", value=len(results_df))
                
                st.markdown("---")
                st.subheader(f"Detailed Analysis Report: {ticker}")
                
                # Stylized dataframe for presentation
                def color_sentiment(val):
                    if val == 'Positive':
                        color = '#d4edda' # Light green
                    elif val == 'Negative':
                        color = '#f8d7da' # Light red
                    else:
                        color = '#f1f3f5' # Light gray
                    return f'background-color: {color}'
                
                styled_df = results_df[['Published', 'Headline', 'Sentiment', 'Positive %', 'Negative %', 'Neutral %']].style.applymap(
                    color_sentiment, subset=['Sentiment']
                ).format({
                    'Positive %': '{:.1f}%',
                    'Negative %': '{:.1f}%',
                    'Neutral %': '{:.1f}%'
                })
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Quick Action recommendations
                st.subheader("Risk Analyst Recommendations")
                if avg_neg > 40:
                    st.error("⚠️ **Action Required:** Highly concentrated negative headlines detected. Recommend escalating for detailed Credit Portfolio Risk evaluation.")
                else:
                    st.success("ℹ️ **No Action Required:** Sentiment signals remain within normal operational thresholds. Monitor weekly.")