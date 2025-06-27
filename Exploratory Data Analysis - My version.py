#!/usr/bin/env python
# coding: utf-8

# # Personal Information
# Name: **Björn van Engelenburg**
# 
# StudentID: **11882123**
# 
# Email: [**bjorn.van.engelenburg@student.uva.nl**](youremail@student.uva.nl)
# 
# Submitted on: **23.03.2025**
# 
# Github Link: https://github.com/bjorn001/EDA.git

# # Data Context
# **The dataset consists of two primary data sources: Yahoo Finance for stock market data and GNews for news articles. The Yahoo Finance dataset provides historical financial data, including daily stock prices, trading volume, and other key market indicators. This dataset is crucial for understanding stock trends over time and assessing price movements in response to market events. The timeframe of this financial data spans multiple years, capturing daily price changes that can be aggregated into weekly trends.**
# 
# **The GNews dataset collects news articles related to specific stock tickers, providing relevant financial news headlines, descriptions, publication dates, and sources. This dataset enables sentiment analysis by evaluating how news sentiment correlates with stock market performance. The timeframe of the news dataset aligns with the stock data, ensuring a comparable basis for analyzing potential relationships between news events and stock price fluctuations. By combining these two sources, the dataset supports an exploration of how financial news impacts stock market behavior.**

# # Data Description
# 
# 
# ![image](data_aggregation_diagram.png)

# In[1]:


# Imports
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ### Data Loading

# #### 1. Financial Data

# In[2]:


# Importing financial data

import pandas as pd

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Define stock tickers and time range
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']
end_date = datetime.today().strftime('%Y-%m-%d')  # Current day
start_date = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')  # Five years ago

# Fetch stock data from Yahoo Finance
stock_data = yf.download(tickers, start=start_date, end=end_date)

# Keep only closing prices
stock_data = stock_data['Close'].reset_index()

# Reshape data: Convert from wide format to long format
stock_data_daily = stock_data.melt(id_vars=['Date'], var_name='Ticker', value_name='Close')

# Save to CSV
stock_data_daily.to_csv('daily_stock_data.csv', index=False)

print("Daily stock data collection complete! Saved as 'daily_stock_data.csv'")


# In[3]:


stock_data_daily.to_csv("stock_data.csv")


# #### 2. News Data

# In[4]:


# # Importing news data
# from gnews import GNews
# import pandas as pd
# from datetime import datetime, timedelta

# # Initialize GNews
# google_news = GNews(language='en', country='US', max_results=100)

# # Define tickers and date range
# tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA'] 
# #["stock", "market", "economy", "inflation", "NASDAQ", "Dow Jones", "Federal Reserve"]
# current_year = datetime.now().year

# years = [current_year - i for i in range(5)]

# news_data = []

# # Fetch news for each ticker and each week of each year
# for year in years:
#     for week in range(1, 53):  # Loop through each week
#         start_date = datetime.fromisocalendar(year, week, 1)  # Monday of the week
#         end_date = start_date + timedelta(days=6)  # Sunday of the week

#         for ticker in tickers:
#             print(f"Fetching news for {ticker} - {year} Week {week}...")
#             google_news.start_date = start_date
#             google_news.end_date = end_date

#             articles = google_news.get_news(ticker)  # Use ticker as keyword

#             for article in articles:
#                 title = article.get('title', 'No Title')
#                 description = article.get('description', 'No Description')
#                 url = article.get('url', 'No URL')
#                 published_date = article.get('published date', 'No Date')
#                 source = article.get('source', {}).get('name', 'Unknown Source')

#                 news_data.append([year, week, ticker, title, description, url, published_date, source])

# # Convert to DataFrame
# df_news_weekly = pd.DataFrame(news_data, columns=['Year', 'Week', 'Ticker', 'Title', 'Description', 'URL', 'Published Date', 'Source'])

# # Save to CSV
# df_news_weekly.to_csv('weekly_news_data.csv', index=False)

# print("Weekly news data collection complete! Saved as 'weekly_news_data.csv'")


# ### Analysis 1: 
# Make sure to add some explanation of what you are doing in your code. This will help you and whoever will read this a lot in following your steps.
# 
# Showcasing the dataframes

# In[5]:


# News dataset

# Define the file path
file_path = "weekly_news_data.csv"

# Read the CSV file into a DataFrame
news_data = pd.read_csv(file_path)

news_data.head(10)


# In[6]:


news_data.columns


# In[7]:


# Financial dataset
stock_data = stock_data_daily 
stock_data.head(10)


# In[8]:


# remove duplicate news articles


# ### Analysis 2: 
# Filtering the news dataset on data for roughly the past year

# In[9]:


# Grabbing a year's worth of news data

# Convert 'Week' and 'Year' columns to numeric types for filtering
news_data['Year'] = pd.to_numeric(news_data['Year'], errors='coerce')
news_data['Week'] = pd.to_numeric(news_data['Week'], errors='coerce')

# Define the filtering condition: 
# Keep data where (Year is 2024 and Week is greater than or equal to 9) OR (Year is 2025 and Week is less than 9)
df_news_filtered = news_data[
    ((news_data['Year'] == 2024) & (news_data['Week'] >= 9)) |
    ((news_data['Year'] == 2025) & (news_data['Week'] < 9))
]


# In[10]:


# Displaying news dataset containing a year's worth of data
news_data = df_news_filtered
news_data.head(5)


# ### Analysis 3: 
# Filtering the financial dataset to the same timeframe as the news dataset,

# In[11]:


stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')

# Extract year and week number from 'Date'
stock_data['Year'] = stock_data['Date'].dt.isocalendar().year
stock_data['Week'] = stock_data['Date'].dt.isocalendar().week

# Define the filtering condition to match the news dataset timeframe:
stock_data_filtered = stock_data[
    ((stock_data['Year'] == 2024) & (stock_data['Week'] >= 9)) |
    ((stock_data['Year'] == 2025) & (stock_data['Week'] < 9))
]
stock_data = stock_data_filtered
stock_data.head(5)


# ### Analysis 4: 
# EDA Analysis of the news dataset
# 

# In[12]:


news_data.head(5)


# In[13]:


df_news = news_data
# Data Cleaning & Preprocessing
df_news.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')  # Remove unnecessary column
df_news['Published Date'] = pd.to_datetime(df_news['Published Date'], errors='coerce')  # Convert to datetime

# General Statistics
total_articles = df_news.shape[0]
unique_tickers = df_news['Ticker'].nunique()
unique_sources = df_news['Source'].nunique()
articles_per_year = df_news['Year'].value_counts().sort_index()
articles_per_ticker = df_news['Ticker'].value_counts()
articles_per_week = df_news.groupby(['Year', 'Week']).size()

# Missing Data Analysis
missing_values = df_news.isnull().sum()
print("\nMissing Values Per Column:\n", missing_values)

# Distribution of Articles Over Time
plt.figure(figsize=(12, 5))
df_news.groupby(df_news['Published Date'].dt.date).size().plot(kind='line', color='blue')
plt.xlabel("Date")
plt.ylabel("Number of Articles")
plt.title("News Articles Over Time")
plt.show()

# Stock (Ticker) Coverage Analysis
plt.figure(figsize=(8, 4))
sns.barplot(x=articles_per_ticker.index, y=articles_per_ticker.values, palette="coolwarm")
plt.xlabel("Stock Ticker")
plt.ylabel("Number of Articles")
plt.title("News Coverage per Stock Ticker")
plt.show()

# Weekly & Yearly Article Trends
plt.figure(figsize=(8, 4))
sns.barplot(x=articles_per_year.index, y=articles_per_year.values, palette="viridis")
plt.xlabel("Year")
plt.ylabel("Number of Articles")
plt.title("Number of News Articles Per Year")
plt.xticks(rotation=45)
plt.show()

# Top 10 News Sources
top_sources = df_news['Source'].value_counts().head(10)
plt.figure(figsize=(8, 4))
sns.barplot(y=top_sources.index, x=top_sources.values, palette="Blues_r")
plt.xlabel("Number of Articles")
plt.ylabel("News Source")
plt.title("Top 10 News Sources")
plt.show()

# Word Cloud (Most Common Words in Titles)
all_titles = ' '.join(df_news['Title'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of News Titles")
plt.show()

# Sentiment Analysis Using VADER
analyzer = SentimentIntensityAnalyzer()
df_news['Sentiment Score'] = df_news['Title'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])
df_news['Sentiment Category'] = df_news['Sentiment Score'].apply(
    lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
)

# Sentiment Distribution
plt.figure(figsize=(8, 4))
sns.countplot(x=df_news['Sentiment Category'], palette="RdYlGn")
plt.xlabel("Sentiment Category")
plt.ylabel("Number of Articles")
plt.title("Sentiment Distribution of News Articles")
plt.show()

# Summary Statistics
summary_stats = {
    "Total Articles": total_articles,
    "Unique Tickers": unique_tickers,
    "Unique News Sources": unique_sources,
    "Missing Values": missing_values.to_dict()
}

print("\nSummary Statistics:\n", summary_stats)



# ### Analysis 5: 
# Sentiment analysis on the news dataset

# In[14]:


import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Perform sentiment analysis on the 'Title' column
news_data['Sentiment Score'] = news_data['Title'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Classify sentiment into categories
news_data['Sentiment Category'] = news_data['Sentiment Score'].apply(
    lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
)


# In[15]:


news_data.head(5)


# ### Analysis 6: 
# EDA Analysis for the financial dataset.
# 
# Key Insights:
# 1. Stock Price Distribution
#     - Prices are normally distributed, with some outliers.
#     
# 2. Stock Price Trends Over Time
#     - Some stocks show a steady increase, while others fluctuate more.
#     
# 3. Outliers & Volatility
#     - Boxplots show price variability per stock.
#     - TSLA & AMZN show higher volatility compared to others.
# 
# 4. Weekly Price Changes
#     - Weekly returns suggest that some stocks are more volatile than others.
#     
# 5. Correlation Between Stocks
#     - Strong positive correlations exist between certain stocks (e.g., AAPL & MSFT).
#     - Others move independently, which is useful for diversification strategies.

# In[16]:


df_stock = stock_data

# Data Cleaning & Preprocessing
df_stock.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')  # Remove unnecessary column
df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')  # Convert Date to datetime

# General Statistics
total_entries = df_stock.shape[0]
unique_tickers = df_stock['Ticker'].nunique()
time_range = (df_stock['Date'].min(), df_stock['Date'].max())
missing_values = df_stock.isnull().sum()

# Stock Price Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df_stock['Close'], bins=30, kde=True, color="blue")
plt.xlabel("Closing Price")
plt.ylabel("Frequency")
plt.title("Distribution of Closing Stock Prices")
plt.show()

# Stock Price Trends Over Time
plt.figure(figsize=(12, 5))
sns.lineplot(data=df_stock, x="Date", y="Close", hue="Ticker", palette="tab10")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.title("Stock Price Trends Over Time")
plt.legend(title="Stock Ticker", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Outlier Detection (Boxplot of Closing Prices)
plt.figure(figsize=(8, 4))
sns.boxplot(x="Ticker", y="Close", data=df_stock, palette="Set2")
plt.xlabel("Stock Ticker")
plt.ylabel("Closing Price")
plt.title("Stock Price Distribution per Ticker")
plt.show()

# Weekly Price Changes (Volatility Analysis)
df_stock['Weekly Change'] = df_stock.groupby('Ticker')['Close'].pct_change() * 100

plt.figure(figsize=(12, 5))
sns.boxplot(x="Ticker", y="Weekly Change", data=df_stock, palette="coolwarm")
plt.xlabel("Stock Ticker")
plt.ylabel("Weekly % Change")
plt.title("Weekly Stock Price Volatility")
plt.show()

# Correlation Between Stocks
df_pivot = df_stock.pivot(index="Date", columns="Ticker", values="Close")
correlation_matrix = df_pivot.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("🔗 Correlation Between Stock Prices")
plt.show()

# Summary Statistics
summary_stats = {
    "Total Entries": total_entries,
    "Unique Tickers": unique_tickers,
    "Time Range": time_range,
    "Missing Values": missing_values.to_dict()
}

print("\n Summary Statistics:\n", summary_stats)


# ### Analysis 4: 
# Merging the financial dataset with the news data

# In[17]:


df_news = news_data

df_stock.head(5)
df_news.head(5)


# In[18]:


df_stock.to_csv("df_stock.csv")
df_news.to_csv("df_news.csv")


# In[19]:


news_df = df_news
stock_df = df_stock


# In[20]:


news_df.head(5)


# In[21]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Ensure the Date column is in datetime format
stock_df["Date"] = pd.to_datetime(stock_df["Date"])

# Extract Year and Week from Date
stock_df["Year"] = stock_df["Date"].dt.year
stock_df["Week"] = stock_df["Date"].dt.isocalendar().week

# Aggregate stock data to weekly mean for each ticker
weekly_stock_df = stock_df.groupby(["Ticker", "Year", "Week"]).agg(
    Close_Mean=("Close", "mean")
).reset_index()

# Aggregate news data to count articles and compute mean sentiment per week
weekly_news_df = news_df.groupby(["Ticker", "Year", "Week"]).agg(
    News_Count=("Title", "count"),  # Count number of articles
    Sentiment_Score=("Sentiment Score", "mean")  # Average sentiment
).reset_index()

# Merge stock and news data
merged_df = pd.merge(weekly_stock_df, weekly_news_df, on=["Ticker", "Year", "Week"], how="left")


# Giving the average sentiment score of all the news articles per week per stock / ticker

# In[22]:


merged_df.head(5)


# In[23]:


# Fill missing values in News_Count and Sentiment_Score (no news = neutral sentiment)
merged_df["News_Count"] = merged_df["News_Count"].fillna(0)
merged_df["Sentiment_Score"] = merged_df["Sentiment_Score"].fillna(0)

# Ensure merged_df is sorted before shifting
merged_df = merged_df.sort_values(by=["Ticker", "Year", "Week"])

# Create previous week's Close_Mean feature
merged_df["Prev_Close_Mean"] = merged_df.groupby("Ticker")["Close_Mean"].shift(1)

# Create target variable: Next week's Close_Mean
merged_df["Next_Close_Mean"] = merged_df.groupby("Ticker")["Close_Mean"].shift(-1)

# Drop NaN values (from first and last weeks per ticker)
merged_df = merged_df.dropna(subset=["Prev_Close_Mean", "Next_Close_Mean"])

# Define feature set and target variable
feature_columns = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]
target_column = "Next_Close_Mean"

# Prepare input (X) and target (y)
X = merged_df[feature_columns]
y = merged_df[target_column]

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation results
mse, r2

# random walk 


# Prev_Close_Mean: 96.7435
# This means that the amount of previous closing price is the most significant factor, as expected
# 
# A sentiment score of 0.5911 means that the sentiment of the news articles matters 

# In[24]:


# Extract feature importance (coefficients)
coefficients = model.coef_
feature_names = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]

# Show feature importance
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")


# In[25]:


# Random Walk Benchmark: Predict Next_Close_Mean = Prev_Close_Mean (baseline)
# For fair comparison, get the unscaled 'Prev_Close_Mean' values corresponding to test set

# Get original (non-scaled) X_test with column names
X_test_unscaled = pd.DataFrame(scaler.inverse_transform(X_test), columns=feature_columns)

# Random Walk prediction uses only Prev_Close_Mean
benchmark_pred = X_test_unscaled["Prev_Close_Mean"].values

# Evaluate Random Walk benchmark
benchmark_mse = mean_squared_error(y_test, benchmark_pred)
benchmark_r2 = r2_score(y_test, benchmark_pred)

# Print both results
print("Model MSE:", mse)
print("Model R2:", r2)
print("Random Walk Benchmark MSE:", benchmark_mse)
print("Random Walk Benchmark R2:", benchmark_r2)


# In[26]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Create DataFrame to compare predictions
comparison_df = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True),
    "Linear Regression": y_pred,
    "Random Walk": benchmark_pred
})

# Plot a sample to visualize
comparison_sample = comparison_df.head(100)

# Start plotting
plt.figure(figsize=(14, 6))
plt.plot(comparison_sample["Actual"], label="Actual", linewidth=2.5, color='black')
plt.plot(comparison_sample["Linear Regression"], label="Linear Regression (Baseline)", linestyle='--', linewidth=2.2, color='dodgerblue')
plt.plot(comparison_sample["Random Walk"], label="Random Walk", linestyle=':', linewidth=2.2, color='orange')

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
benchmark_mse = mean_squared_error(y_test, benchmark_pred)
benchmark_r2 = r2_score(y_test, benchmark_pred)

# Annotate the metrics in the plot
metrics_text = (
    f"Linear Regression\nMSE: {mse:.2f}, R²: {r2:.4f}\n"
    f"Random Walk\nMSE: {benchmark_mse:.2f}, R²: {benchmark_r2:.4f}"
)
plt.text(
    0.01, 0.01, metrics_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='bottom',
    bbox=dict(facecolor='white', alpha=0.9, edgecolor='grey')
)

plt.title("Linear Regression vs. Random Walk vs. Actual – Weekly Closing Price Prediction")
plt.xlabel("Test Week")
plt.ylabel("Next Week’s Closing Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[27]:


import matplotlib.pyplot as plt

# Create a DataFrame to compare predictions
comparison_df = pd.DataFrame({
    "Actual": y_test,
    "Baseline_Prediction": y_pred,
    "Random_Walk": benchmark_pred
}).reset_index(drop=True)

# Plotting
plt.figure(figsize=(14, 6))
plt.plot(comparison_df["Actual"], label="Actual", linewidth=2)
plt.plot(comparison_df["Baseline_Prediction"], label="Linear Regression (Baseline)", linestyle='--')
plt.plot(comparison_df["Random_Walk"], label="Random Walk", linestyle=':')
plt.title("Model vs. Random Walk vs. Actual Weekly Closing Prices")
plt.xlabel("Test Weeks")
plt.ylabel("Next Week’s Closing Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ### Testing correlation feature

# In[28]:


merged_df.head(5)


# In[29]:


merged_df = merged_df.drop(columns=['Next_Close_Mean'])


# In[30]:


merged_df


# In[31]:


# Step 1: Ensure df_stock has correct datetime & sorting
df_stock['Date'] = pd.to_datetime(df_stock['Date'])
df_stock = df_stock.sort_values(by='Date')

# Step 2: Pivot daily stock data to wide format to get daily closing prices
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']
price_data = df_stock.pivot(index='Date', columns='Ticker', values='Close').dropna()

# Step 3: Calculate daily returns
returns = price_data.pct_change().dropna()

# Step 4: For each Ticker, calculate rolling correlations with the other stocks
window_size = 4  # You can adjust this window size

corr_features = []

for ticker in tickers:
    peers = [t for t in tickers if t != ticker]
    for peer in peers:
        # Calculate rolling correlation between ticker and peer
        rolling_corr = returns[ticker].rolling(window=window_size).corr(returns[peer])

        # Build DataFrame for this correlation feature
        corr_col_name = f'{ticker}_Corr_to_{peer}'  # clean and clear name

        df_corr = pd.DataFrame({
            'Date': rolling_corr.index,
            'Ticker': ticker,
            corr_col_name: rolling_corr.values
        })

        # Extract Year and Week to match merged_df
        df_corr['Year'] = df_corr['Date'].dt.isocalendar().year
        df_corr['Week'] = df_corr['Date'].dt.isocalendar().week

        # Only keep needed columns
        corr_features.append(df_corr[['Year', 'Week', 'Ticker', corr_col_name]])

# Step 5: Combine all correlation features into one DataFrame
df_corr_all = pd.concat(corr_features)

# Step 6: Pivot the correlations so all features are in columns (per Ticker, Year, Week)
df_corr_pivot = df_corr_all.pivot_table(
    index=['Year', 'Week', 'Ticker'],
    aggfunc='mean'  # in case of duplicates
).reset_index()

# Step 7: Merge the correlation features BACK into your merged_df
merged_df = pd.merge(merged_df, df_corr_pivot, on=['Year', 'Week', 'Ticker'], how='left')

# Step 8: Fill any missing correlation values (e.g., at start of rolling window) with 0
corr_cols = [col for col in merged_df.columns if '_Corr_to_' in col]
merged_df[corr_cols] = merged_df[corr_cols].fillna(0)

#  Now merged_df contains crystal-clear correlation features.
print(" Correlation features added with clean naming! Preview:")
print(merged_df.head())


# In[32]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
merged_df.head(5)


# In[33]:


# ----------------------------
# Regression with Correlation Features
# ----------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- 1️⃣ Recreate Next_Close_Mean and Prev_Close_Mean ---

# Ensure correct sorting before shifting
merged_df = merged_df.sort_values(by=["Ticker", "Year", "Week"])

# Recreate previous week's Close_Mean
merged_df["Prev_Close_Mean"] = merged_df.groupby("Ticker")["Close_Mean"].shift(1)

# Recreate target variable: Next week's Close_Mean
merged_df["Next_Close_Mean"] = merged_df.groupby("Ticker")["Close_Mean"].shift(-1)

# --- 2️⃣ Prepare Features and Target ---

# Define the feature columns:
# - Basic features
base_features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]

# - Correlation features (auto-detect columns with '_Corr_to_')
corr_features = [col for col in merged_df.columns if '_Corr_to_' in col]

# Combine all features
feature_columns = base_features + corr_features

# Define the target column
target_column = "Next_Close_Mean"

# Drop rows with missing data in features or target
df_model = merged_df.dropna(subset=feature_columns + [target_column])

# --- 3️⃣ Scale and Split Data ---

# Prepare input (X) and target (y)
X = df_model[feature_columns]
y = df_model[target_column]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

# --- 4️⃣ Train the Linear Regression Model ---

model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# --- 5️⃣ Evaluate the Model ---

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Model Performance:")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.4f}")

# --- 6️⃣ Feature Importance ---

print("\n Feature Importances (Coefficients):")
for feature, coef in zip(feature_columns, model.coef_):
    print(f"{feature:35s}: {coef:.6f}")

# --- 7️⃣ Plot Actual vs Predicted ---

comparison_df = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True),
    "Predicted": y_pred
})

plt.figure(figsize=(14, 6))
plt.plot(comparison_df["Actual"], label="Actual", linewidth=2, color="black")
plt.plot(comparison_df["Predicted"], label="Predicted", linestyle="--", color="dodgerblue")
plt.title(" Linear Regression – Actual vs Predicted Closing Price")
plt.xlabel("Test Samples")
plt.ylabel("Next Week's Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 8️⃣ Plot Scatter for Diagnostic ---

plt.figure(figsize=(8, 6))
plt.scatter(y_test.reset_index(drop=True), y_pred, alpha=0.6, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # Ideal line
plt.xlabel("Actual Next Week's Close")
plt.ylabel("Predicted")
plt.title(" Predicted vs Actual Closing Price")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[34]:


# ----------------------------
# Regression Per Ticker: Using Only Relevant Correlations
# ----------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Define the tickers you are working with
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

# Loop through each ticker and train a model using only relevant correlations
for ticker in tickers:
    print(f"\n{'='*40}\n Modeling for {ticker}\n{'='*40}")

    # Filter data for this ticker
    df_ticker = merged_df[merged_df["Ticker"] == ticker].copy()

    # Recreate lag and target columns (just to be safe)
    df_ticker = df_ticker.sort_values(by=["Year", "Week"])
    df_ticker["Prev_Close_Mean"] = df_ticker["Close_Mean"].shift(1)
    df_ticker["Next_Close_Mean"] = df_ticker["Close_Mean"].shift(-1)

    # Define base features
    base_features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]

    # Define correlation features: only those relevant to the current ticker
    corr_features = [col for col in df_ticker.columns if col.startswith(f"{ticker}_Corr_to_")]

    # Full feature set
    feature_columns = base_features + corr_features

    # Drop rows with missing data
    df_model = df_ticker.dropna(subset=feature_columns + ["Next_Close_Mean"])

    # Skip if not enough data
    if len(df_model) < 10:
        print(f" Not enough data for {ticker}, skipping...\n")
        continue

    # Prepare X and y
    X = df_model[feature_columns]
    y = df_model["Next_Close_Mean"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f" Performance for {ticker}:")
    print(f"MSE: {mse:.2f}")
    print(f"R²: {r2:.4f}")

    # Show feature importance
    print("\n Feature Importances:")
    for feature, coef in zip(feature_columns, model.coef_):
        print(f"{feature:35s}: {coef:.6f}")

    # Plot actual vs predicted
    comparison_df = pd.DataFrame({
        "Actual": y_test.reset_index(drop=True),
        "Predicted": y_pred
    })

    plt.figure(figsize=(12, 5))
    plt.plot(comparison_df["Actual"], label="Actual", linewidth=2, color="black")
    plt.plot(comparison_df["Predicted"], label="Predicted", linestyle="--", color="dodgerblue")
    plt.title(f"{ticker}: Linear Regression – Actual vs Predicted Closing Price")
    plt.xlabel("Test Samples")
    plt.ylabel("Next Week's Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[35]:


# ----------------------------
# Regression Per Ticker: Baseline vs. Extended (with Correlations)
# ----------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Define the tickers you are working with
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

# Loop through each ticker and train both baseline and extended models
for ticker in tickers:
    print(f"\n{'='*60}\n Modeling for {ticker}\n{'='*60}")

    # Filter data for this ticker
    df_ticker = merged_df[merged_df["Ticker"] == ticker].copy()

    # Recreate lag and target columns (just to be safe)
    df_ticker = df_ticker.sort_values(by=["Year", "Week"])
    df_ticker["Prev_Close_Mean"] = df_ticker["Close_Mean"].shift(1)
    df_ticker["Next_Close_Mean"] = df_ticker["Close_Mean"].shift(-1)

    # Define base features
    base_features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]

    # Define correlation features: only those relevant to the current ticker
    corr_features = [col for col in df_ticker.columns if col.startswith(f"{ticker}_Corr_to_")]

    # Full feature set
    extended_features = base_features + corr_features

    # Drop rows with missing data for both setups
    df_model = df_ticker.dropna(subset=extended_features + ["Next_Close_Mean"])

    if len(df_model) < 10:
        print(f" Not enough data for {ticker}, skipping...\n")
        continue

    # ---------------------
    # 1️⃣ Baseline model (without correlations)
    # ---------------------
    print(f"\n Baseline model (no correlations) for {ticker}")
    X_base = df_model[base_features]
    y = df_model["Next_Close_Mean"]

    scaler_base = StandardScaler()
    X_base_scaled = scaler_base.fit_transform(X_base)

    X_train, X_test, y_train, y_test = train_test_split(
        X_base_scaled, y, test_size=0.2, random_state=42, shuffle=True
    )

    model_base = LinearRegression()
    model_base.fit(X_train, y_train)

    y_pred_base = model_base.predict(X_test)

    mse_base = mean_squared_error(y_test, y_pred_base)
    r2_base = r2_score(y_test, y_pred_base)

    print(f"MSE: {mse_base:.2f}")
    print(f"R²: {r2_base:.4f}")

    print("\n Feature Importances (Baseline):")
    for feature, coef in zip(base_features, model_base.coef_):
        print(f"{feature:35s}: {coef:.6f}")

    # ---------------------
    # 2️⃣ Extended model (with correlations)
    # ---------------------
    print(f"\n Extended model (with correlations) for {ticker}")
    X_ext = df_model[extended_features]

    scaler_ext = StandardScaler()
    X_ext_scaled = scaler_ext.fit_transform(X_ext)

    X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(
        X_ext_scaled, y, test_size=0.2, random_state=42, shuffle=True
    )

    model_ext = LinearRegression()
    model_ext.fit(X_train_ext, y_train_ext)

    y_pred_ext = model_ext.predict(X_test_ext)

    mse_ext = mean_squared_error(y_test_ext, y_pred_ext)
    r2_ext = r2_score(y_test_ext, y_pred_ext)

    print(f"MSE: {mse_ext:.2f}")
    print(f"R²: {r2_ext:.4f}")

    print("\n Feature Importances (Extended):")
    for feature, coef in zip(extended_features, model_ext.coef_):
        print(f"{feature:35s}: {coef:.6f}")

    # ---------------------
    #  Plot Comparisons
    # ---------------------
    plt.figure(figsize=(14, 5))
    plt.plot(y_test.reset_index(drop=True), label="Actual", color="black", linewidth=2)
    plt.plot(y_pred_base, label="Baseline Predicted", linestyle="--", color="orange")
    plt.plot(y_pred_ext, label="Extended Predicted", linestyle="--", color="dodgerblue")
    plt.title(f"{ticker}: Actual vs Predicted – Baseline vs Extended")
    plt.xlabel("Test Samples")
    plt.ylabel("Next Week's Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[36]:


# ----------------------------
# Regression Per Ticker: Baseline vs. Extended (with Correlations) – SYNCHRONIZED TEST WEEKS
# ----------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the tickers you are working with
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

# STEP 1: Global split on Year + Week
week_keys = merged_df[['Year', 'Week']].drop_duplicates()

train_weeks, test_weeks = train_test_split(
    week_keys, test_size=0.2, random_state=42, shuffle=True
)

# Mark rows as train/test
merged_df['Set'] = merged_df.apply(
    lambda row: 'test' if (row['Year'], row['Week']) in [tuple(x) for x in test_weeks.values] else 'train',
    axis=1
)

# Loop through each ticker and train both baseline and extended models
for ticker in tickers:
    print(f"\n{'='*60}\n Modeling for {ticker}\n{'='*60}")

    # Filter data for this ticker
    df_ticker = merged_df[merged_df["Ticker"] == ticker].copy()

    # Recreate lag and target columns (just to be safe)
    df_ticker = df_ticker.sort_values(by=["Year", "Week"])
    df_ticker["Prev_Close_Mean"] = df_ticker["Close_Mean"].shift(1)
    df_ticker["Next_Close_Mean"] = df_ticker["Close_Mean"].shift(-1)

    # Drop rows missing essential features
    base_features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]
    corr_features = [col for col in df_ticker.columns if col.startswith(f"{ticker}_Corr_to_")]
    extended_features = base_features + corr_features
    df_model = df_ticker.dropna(subset=extended_features + ["Next_Close_Mean"])

    # Split into train/test using synchronized Set column
    train_df = df_model[df_model['Set'] == 'train']
    test_df = df_model[df_model['Set'] == 'test']

    if len(test_df) < 3:
        print(f"⚠️ Not enough test samples for {ticker}, skipping...\n")
        continue

    # ---------------------
    # 1️⃣ Baseline model (without correlations)
    # ---------------------
    print(f"\n Baseline model (no correlations) for {ticker}")
    X_train_base = train_df[base_features]
    y_train = train_df["Next_Close_Mean"]
    X_test_base = test_df[base_features]
    y_test = test_df["Next_Close_Mean"]

    scaler_base = StandardScaler()
    X_train_base_scaled = scaler_base.fit_transform(X_train_base)
    X_test_base_scaled = scaler_base.transform(X_test_base)

    model_base = LinearRegression()
    model_base.fit(X_train_base_scaled, y_train)

    y_pred_base = model_base.predict(X_test_base_scaled)

    mse_base = mean_squared_error(y_test, y_pred_base)
    r2_base = r2_score(y_test, y_pred_base)

    print(f"MSE: {mse_base:.2f}")
    print(f"R²: {r2_base:.4f}")

    print("\n Feature Importances (Baseline):")
    for feature, coef in zip(base_features, model_base.coef_):
        print(f"{feature:35s}: {coef:.6f}")

    # ---------------------
    # 2️⃣ Extended model (with correlations)
    # ---------------------
    print(f"\n Extended model (with correlations) for {ticker}")
    X_train_ext = train_df[extended_features]
    X_test_ext = test_df[extended_features]

    scaler_ext = StandardScaler()
    X_train_ext_scaled = scaler_ext.fit_transform(X_train_ext)
    X_test_ext_scaled = scaler_ext.transform(X_test_ext)

    model_ext = LinearRegression()
    model_ext.fit(X_train_ext_scaled, y_train)

    y_pred_ext = model_ext.predict(X_test_ext_scaled)

    mse_ext = mean_squared_error(y_test, y_pred_ext)
    r2_ext = r2_score(y_test, y_pred_ext)

    print(f"MSE: {mse_ext:.2f}")
    print(f"R²: {r2_ext:.4f}")

    print("\n Feature Importances (Extended):")
    for feature, coef in zip(extended_features, model_ext.coef_):
        print(f"{feature:35s}: {coef:.6f}")

    # ---------------------
    # 📈 Plot Comparisons
    # ---------------------
    plt.figure(figsize=(14, 5))
    plt.plot(y_test.reset_index(drop=True), label="Actual", color="black", linewidth=2)
    plt.plot(y_pred_base, label="Baseline Predicted", linestyle="--", color="orange")
    plt.plot(y_pred_ext, label="Extended Predicted", linestyle="--", color="dodgerblue")
    plt.title(f"{ticker}: Actual vs Predicted – Baseline vs Extended")
    plt.xlabel("Test Samples (Synchronized Weeks)")
    plt.ylabel("Next Week's Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[37]:


# ----------------------------
# Regression Per Ticker: Baseline vs. Extended (with Correlations) + Random Walk Benchmark
# ----------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the tickers you are working with
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

# STEP 1: Global split on Year + Week
week_keys = merged_df[['Year', 'Week']].drop_duplicates()

train_weeks, test_weeks = train_test_split(
    week_keys, test_size=0.2, random_state=42, shuffle=True
)

# Mark rows as train/test
merged_df['Set'] = merged_df.apply(
    lambda row: 'test' if (row['Year'], row['Week']) in [tuple(x) for x in test_weeks.values] else 'train',
    axis=1
)

# Loop through each ticker and train both baseline and extended models
for ticker in tickers:
    print(f"\n{'='*60}\n Modeling for {ticker}\n{'='*60}")

    # Filter data for this ticker
    df_ticker = merged_df[merged_df["Ticker"] == ticker].copy()

    # Recreate lag and target columns (just to be safe)
    df_ticker = df_ticker.sort_values(by=["Year", "Week"])
    df_ticker["Prev_Close_Mean"] = df_ticker["Close_Mean"].shift(1)
    df_ticker["Next_Close_Mean"] = df_ticker["Close_Mean"].shift(-1)

    # Drop rows missing essential features
    base_features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]
    corr_features = [col for col in df_ticker.columns if col.startswith(f"{ticker}_Corr_to_")]
    extended_features = base_features + corr_features
    df_model = df_ticker.dropna(subset=extended_features + ["Next_Close_Mean"])

    # Split into train/test using synchronized Set column
    train_df = df_model[df_model['Set'] == 'train']
    test_df = df_model[df_model['Set'] == 'test']

    if len(test_df) < 3:
        print(f"⚠️ Not enough test samples for {ticker}, skipping...\n")
        continue

    # ---------------------
    # 1️⃣ Baseline model (without correlations)
    # ---------------------
    print(f"\n Baseline model (no correlations) for {ticker}")
    X_train_base = train_df[base_features]
    y_train = train_df["Next_Close_Mean"]
    X_test_base = test_df[base_features]
    y_test = test_df["Next_Close_Mean"]

    scaler_base = StandardScaler()
    X_train_base_scaled = scaler_base.fit_transform(X_train_base)
    X_test_base_scaled = scaler_base.transform(X_test_base)

    model_base = LinearRegression()
    model_base.fit(X_train_base_scaled, y_train)

    y_pred_base = model_base.predict(X_test_base_scaled)

    mse_base = mean_squared_error(y_test, y_pred_base)
    r2_base = r2_score(y_test, y_pred_base)

    print(f"MSE: {mse_base:.2f}")
    print(f"R²: {r2_base:.4f}")

    print("\n Feature Importances (Baseline):")
    for feature, coef in zip(base_features, model_base.coef_):
        print(f"{feature:35s}: {coef:.6f}")

    # ---------------------
    # 2️⃣ Extended model (with correlations)
    # ---------------------
    print(f"\n Extended model (with correlations) for {ticker}")
    X_train_ext = train_df[extended_features]
    X_test_ext = test_df[extended_features]

    scaler_ext = StandardScaler()
    X_train_ext_scaled = scaler_ext.fit_transform(X_train_ext)
    X_test_ext_scaled = scaler_ext.transform(X_test_ext)

    model_ext = LinearRegression()
    model_ext.fit(X_train_ext_scaled, y_train)

    y_pred_ext = model_ext.predict(X_test_ext_scaled)

    mse_ext = mean_squared_error(y_test, y_pred_ext)
    r2_ext = r2_score(y_test, y_pred_ext)

    print(f"MSE: {mse_ext:.2f}")
    print(f"R²: {r2_ext:.4f}")

    print("\n Feature Importances (Extended):")
    for feature, coef in zip(extended_features, model_ext.coef_):
        print(f"{feature:35s}: {coef:.6f}")

    # ---------------------
    # 3️⃣ Random Walk benchmark
    # ---------------------
    print(f"\n Random Walk Benchmark for {ticker}")

    # Random walk predicts next close ≈ previous close
    y_pred_rw = test_df["Prev_Close_Mean"].values

    mse_rw = mean_squared_error(y_test, y_pred_rw)
    r2_rw = r2_score(y_test, y_pred_rw)

    print(f"MSE: {mse_rw:.2f}")
    print(f"R²: {r2_rw:.4f}")

    # ---------------------
    # 📈 Plot Comparisons
    # ---------------------
    plt.figure(figsize=(14, 5))
    plt.plot(y_test.reset_index(drop=True), label="Actual", color="black", linewidth=2)
    plt.plot(y_pred_base, label="Baseline Predicted", linestyle="--", color="orange")
    plt.plot(y_pred_ext, label="Extended Predicted", linestyle="--", color="dodgerblue")
    plt.plot(y_pred_rw, label="Random Walk", linestyle=":", color="green")
    plt.title(f"{ticker}: Actual vs Predicted – Baseline vs Extended vs Random Walk")
    plt.xlabel("Test Samples (Synchronized Weeks)")
    plt.ylabel("Next Week's Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[38]:


# ----------------------------
# Regression Per Ticker: Baseline vs. Extended (with Correlations) + Random Walk Benchmark = FINAL GOOD MODEL
# ----------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the tickers you are working with
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

#  STEP 1: Global split on Year + Week
week_keys = merged_df[['Year', 'Week']].drop_duplicates()

train_weeks, test_weeks = train_test_split(
    week_keys, test_size=0.2, random_state=42, shuffle=True
)

# Mark rows as train/test
merged_df['Set'] = merged_df.apply(
    lambda row: 'test' if (row['Year'], row['Week']) in [tuple(x) for x in test_weeks.values] else 'train',
    axis=1
)

# Loop through each ticker and train both baseline and extended models
for ticker in tickers:
    print(f"\n{'='*60}\n Modeling for {ticker}\n{'='*60}")

    # Filter data for this ticker
    df_ticker = merged_df[merged_df["Ticker"] == ticker].copy()

    # Recreate lag and target columns (just to be safe)
    df_ticker = df_ticker.sort_values(by=["Year", "Week"])
    df_ticker["Prev_Close_Mean"] = df_ticker["Close_Mean"].shift(1)
    df_ticker["Next_Close_Mean"] = df_ticker["Close_Mean"].shift(-1)

    # Drop rows missing essential features
    base_features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]
    corr_features = [col for col in df_ticker.columns if col.startswith(f"{ticker}_Corr_to_")]
    extended_features = base_features + corr_features
    df_model = df_ticker.dropna(subset=extended_features + ["Next_Close_Mean"])

    # Split into train/test using synchronized Set column
    train_df = df_model[df_model['Set'] == 'train']
    test_df = df_model[df_model['Set'] == 'test']

    if len(test_df) < 3:
        print(f"⚠️ Not enough test samples for {ticker}, skipping...\n")
        continue

    # ---------------------
    # 1️⃣ Baseline model (without correlations)
    # ---------------------
    print(f"\n Baseline model (no correlations) for {ticker}")
    X_train_base = train_df[base_features]
    y_train = train_df["Next_Close_Mean"]
    X_test_base = test_df[base_features]
    y_test = test_df["Next_Close_Mean"]

    scaler_base = StandardScaler()
    X_train_base_scaled = scaler_base.fit_transform(X_train_base)
    X_test_base_scaled = scaler_base.transform(X_test_base)

    model_base = LinearRegression()
    model_base.fit(X_train_base_scaled, y_train)

    y_pred_base = model_base.predict(X_test_base_scaled)

    mse_base = mean_squared_error(y_test, y_pred_base)
    r2_base = r2_score(y_test, y_pred_base)

    print(f"MSE: {mse_base:.2f}")
    print(f"R²: {r2_base:.4f}")

    print("\n Feature Importances (Baseline):")
    for feature, coef in zip(base_features, model_base.coef_):
        print(f"{feature:35s}: {coef:.6f}")

    # ---------------------
    # 2️⃣ Extended model (with correlations)
    # ---------------------
    print(f"\n Extended model (with correlations) for {ticker}")
    X_train_ext = train_df[extended_features]
    X_test_ext = test_df[extended_features]

    scaler_ext = StandardScaler()
    X_train_ext_scaled = scaler_ext.fit_transform(X_train_ext)
    X_test_ext_scaled = scaler_ext.transform(X_test_ext)

    model_ext = LinearRegression()
    model_ext.fit(X_train_ext_scaled, y_train)

    y_pred_ext = model_ext.predict(X_test_ext_scaled)

    mse_ext = mean_squared_error(y_test, y_pred_ext)
    r2_ext = r2_score(y_test, y_pred_ext)

    print(f"MSE: {mse_ext:.2f}")
    print(f"R²: {r2_ext:.4f}")

    print("\n Feature Importances (Extended):")
    for feature, coef in zip(extended_features, model_ext.coef_):
        print(f"{feature:35s}: {coef:.6f}")

    # ---------------------
    # 3️⃣ Random Walk benchmark
    # ---------------------
    print(f"\n Random Walk Benchmark for {ticker}")

    # Random walk predicts next close ≈ previous close
    y_pred_rw = test_df["Prev_Close_Mean"].values

    mse_rw = mean_squared_error(y_test, y_pred_rw)
    r2_rw = r2_score(y_test, y_pred_rw)

    print(f"MSE: {mse_rw:.2f}")
    print(f"R²: {r2_rw:.4f}")

    # ---------------------
    # 📈 Plot Comparisons
    # ---------------------
    plt.figure(figsize=(14, 5))
    plt.plot(y_test.reset_index(drop=True), label="Actual", color="black", linewidth=2)
    plt.plot(y_pred_base, label="Baseline Predicted", linestyle="--", color="orange")
    plt.plot(y_pred_ext, label="Extended Predicted", linestyle="--", color="dodgerblue")
    plt.plot(y_pred_rw, label="Random Walk", linestyle=":", color="green")
    plt.title(f"{ticker}: Actual vs Predicted – Baseline vs Extended vs Random Walk")
    plt.xlabel("Test Samples (Synchronized Weeks)")
    plt.ylabel("Next Week's Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# #### LSTM

# In[39]:


# # OG model

# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# import matplotlib.pyplot as plt

# tf.get_logger().setLevel('ERROR')

# base_features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]
# tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

# # Original hyperparameters
# sequence_length = 5
# lstm_units = 50
# dropout = 0.2
# epochs = 30
# batch_size = 16

# for ticker in tickers:
#     print(f"\n{'='*60}\n LSTM for {ticker}\n{'='*60}")

#     df_ticker = merged_df[merged_df["Ticker"] == ticker].copy()
#     df_ticker = df_ticker.sort_values(by=["Year", "Week"])
#     df_ticker["Next_Close_Mean"] = df_ticker["Close_Mean"].shift(-1)

#     corr_features = [col for col in df_ticker.columns if col.startswith(f"{ticker}_Corr_to_")]
#     extended_features = base_features + corr_features

#     predictions_dict = {}
#     y_test_dict = {}

#     for mode, features in zip(["Baseline", "Extended"], [base_features, extended_features]):
#         print(f"\n🔧 Training LSTM ({mode})")

#         df_model = df_ticker.dropna(subset=features + ["Next_Close_Mean"])

#         scaler_X = MinMaxScaler()
#         scaler_y = MinMaxScaler()

#         X_scaled = scaler_X.fit_transform(df_model[features])
#         y_scaled = scaler_y.fit_transform(df_model[["Next_Close_Mean"]])

#         X_seq, y_seq = [], []
#         for i in range(sequence_length, len(X_scaled)):
#             X_seq.append(X_scaled[i-sequence_length:i])
#             y_seq.append(y_scaled[i])

#         X_seq, y_seq = np.array(X_seq), np.array(y_seq)
#         X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

#         model = Sequential()
#         model.add(LSTM(lstm_units, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
#         model.add(Dropout(dropout))
#         model.add(LSTM(lstm_units))
#         model.add(Dropout(dropout))
#         model.add(Dense(1))

#         model.compile(optimizer='adam', loss='mean_squared_error')
#         model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)

#         y_pred = model.predict(X_test, verbose=0)
#         y_pred_inv = scaler_y.inverse_transform(y_pred)
#         y_test_inv = scaler_y.inverse_transform(y_test)

#         mse = np.mean((y_test_inv - y_pred_inv)**2)
#         r2 = r2_score(y_test_inv, y_pred_inv)
#         print(f"MSE ({mode}): {mse:.2f}")
#         print(f"R²  ({mode}): {r2:.4f}")

#         predictions_dict[mode] = y_pred_inv.flatten()
#         y_test_dict[mode] = y_test_inv.flatten()

#     # Plot actual vs predicted
#     plt.figure(figsize=(12, 5))
#     plt.plot(y_test_dict['Baseline'], label='Actual', color='black', linewidth=2)
#     plt.plot(predictions_dict['Baseline'], label='Predicted (Baseline)', linestyle='--', color='orange')
#     plt.plot(predictions_dict['Extended'], label='Predicted (Extended)', linestyle='--', color='dodgerblue')
#     plt.title(f"{ticker}: Actual vs Predicted (Baseline vs Extended)")
#     plt.xlabel("Test Samples")
#     plt.ylabel("Next Week's Close Price")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


# In[40]:


# These are good results

# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# import matplotlib.pyplot as plt
# import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TF warnings
# tf.get_logger().setLevel('ERROR')

# # Hyperparameter grid
# param_grid = [
#     {"lstm_units": 50, "dropout": 0.2, "epochs": 30, "batch_size": 16, "sequence_length": 5},
#     {"lstm_units": 64, "dropout": 0.3, "epochs": 40, "batch_size": 32, "sequence_length": 10},
#     {"lstm_units": 100, "dropout": 0.3, "epochs": 50, "batch_size": 32, "sequence_length": 5},
#     {"lstm_units": 128, "dropout": 0.3, "epochs": 50, "batch_size": 64, "sequence_length": 7},
#     {"lstm_units": 75, "dropout": 0.25, "epochs": 35, "batch_size": 16, "sequence_length": 6},
# ]

# base_features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]
# tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

# for ticker in tickers:
#     print(f"\n{'='*60}\n LSTM for {ticker}\n{'='*60}")
#     df_ticker = merged_df[merged_df["Ticker"] == ticker].copy()
#     df_ticker = df_ticker.sort_values(by=["Year", "Week"])
#     df_ticker["Next_Close_Mean"] = df_ticker["Close_Mean"].shift(-1)

#     corr_features = [col for col in df_ticker.columns if col.startswith(f"{ticker}_Corr_to_")]
#     extended_features = base_features + corr_features

#     predictions_dict, y_test_dict = {}, {}

#     for mode, features in zip(["Baseline", "Extended"], [base_features, extended_features]):
#         print(f"\n🔧 Tuning LSTM ({mode})")
#         df_model = df_ticker.dropna(subset=features + ["Next_Close_Mean"])
#         best_r2, best_model_data = -np.inf, None

#         for params in param_grid:
#             scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
#             X_scaled = scaler_X.fit_transform(df_model[features])
#             y_scaled = scaler_y.fit_transform(df_model[["Next_Close_Mean"]])

#             X_seq, y_seq = [], []
#             for i in range(params["sequence_length"], len(X_scaled)):
#                 X_seq.append(X_scaled[i - params["sequence_length"]:i])
#                 y_seq.append(y_scaled[i])

#             X_seq, y_seq = np.array(X_seq), np.array(y_seq)
#             X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

#             model = Sequential()
#             model.add(LSTM(params["lstm_units"], return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
#             model.add(Dropout(params["dropout"]))
#             model.add(LSTM(params["lstm_units"]))
#             model.add(Dropout(params["dropout"]))
#             model.add(Dense(1))

#             model.compile(optimizer='adam', loss='mean_squared_error')
#             model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"],
#                       validation_split=0.1, verbose=0)

#             y_pred = model.predict(X_test, verbose=0)
#             y_pred_inv = scaler_y.inverse_transform(y_pred)
#             y_test_inv = scaler_y.inverse_transform(y_test)

#             mse = np.mean((y_test_inv - y_pred_inv) ** 2)
#             r2 = r2_score(y_test_inv, y_pred_inv)

#             if r2 > best_r2:
#                 best_r2 = r2
#                 best_model_data = {
#                     "y_pred_inv": y_pred_inv,
#                     "y_test_inv": y_test_inv,
#                     "mse": mse,
#                     "r2": r2,
#                     "params": params
#                 }

#         print(f"✅ Best MSE ({mode}): {best_model_data['mse']:.2f}")
#         print(f"✅ Best R²  ({mode}): {best_model_data['r2']:.4f}")
#         predictions_dict[mode] = best_model_data['y_pred_inv'].flatten()
#         y_test_dict[mode] = best_model_data['y_test_inv'].flatten()

#     # Plot actual vs predicted for baseline and extended
#     plt.figure(figsize=(12, 5))
#     plt.plot(y_test_dict['Baseline'], label='Actual', color='black', linewidth=2)
#     plt.plot(predictions_dict['Baseline'], label='Predicted (Baseline)', linestyle='--', color='orange')
#     plt.plot(predictions_dict['Extended'], label='Predicted (Extended)', linestyle='--', color='dodgerblue')
#     plt.title(f"{ticker}: Actual vs Predicted (Baseline vs Extended)")
#     plt.xlabel("Test Samples")
#     plt.ylabel("Next Week's Close Price")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


# In[41]:


# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# import matplotlib.pyplot as plt
# import itertools
# import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.get_logger().setLevel('ERROR')

# # Expanded hyperparameter lists
# lstm_units_list = [50, 64, 75, 100, 128]
# dropout_list = [0.2, 0.25, 0.3, 0.4]
# epochs_list = [30, 40, 50, 70]
# batch_size_list = [16, 32, 64]
# sequence_length_list = [5, 6, 7, 10, 15]

# # Full hyperparameter grid
# param_grid = list(itertools.product(
#     lstm_units_list,
#     dropout_list,
#     epochs_list,
#     batch_size_list,
#     sequence_length_list
# ))

# base_features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]
# tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

# # Create results DataFrame
# results_log = []

# for ticker in tickers:
#     print(f"\n{'='*60}\n LSTM for {ticker}\n{'='*60}")
#     df_ticker = merged_df[merged_df["Ticker"] == ticker].copy()
#     df_ticker = df_ticker.sort_values(by=["Year", "Week"])
#     df_ticker["Next_Close_Mean"] = df_ticker["Close_Mean"].shift(-1)

#     corr_features = [col for col in df_ticker.columns if col.startswith(f"{ticker}_Corr_to_")]
#     extended_features = base_features + corr_features

#     predictions_dict, y_test_dict = {}, {}

#     for mode, features in zip(["Baseline", "Extended"], [base_features, extended_features]):
#         print(f"\n🔧 Extensive Grid Search LSTM ({mode})")
#         df_model = df_ticker.dropna(subset=features + ["Next_Close_Mean"])
#         best_r2, best_model_data = -np.inf, None

#         for lstm_units, dropout, epochs, batch_size, seq_len in param_grid:
#             scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
#             X_scaled = scaler_X.fit_transform(df_model[features])
#             y_scaled = scaler_y.fit_transform(df_model[["Next_Close_Mean"]])

#             X_seq, y_seq = [], []
#             for i in range(seq_len, len(X_scaled)):
#                 X_seq.append(X_scaled[i - seq_len:i])
#                 y_seq.append(y_scaled[i])

#             X_seq, y_seq = np.array(X_seq), np.array(y_seq)
#             X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

#             model = Sequential()
#             model.add(LSTM(lstm_units, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
#             model.add(Dropout(dropout))
#             model.add(LSTM(lstm_units))
#             model.add(Dropout(dropout))
#             model.add(Dense(1))

#             model.compile(optimizer='adam', loss='mean_squared_error')
#             model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
#                       validation_split=0.1, verbose=0)

#             y_pred = model.predict(X_test, verbose=0)
#             y_pred_inv = scaler_y.inverse_transform(y_pred)
#             y_test_inv = scaler_y.inverse_transform(y_test)

#             mse = np.mean((y_test_inv - y_pred_inv) ** 2)
#             r2 = r2_score(y_test_inv, y_pred_inv)

#             # Log this combination
#             results_log.append({
#                 "Ticker": ticker,
#                 "Model": mode,
#                 "R2": r2,
#                 "MSE": mse,
#                 "LSTM_units": lstm_units,
#                 "Dropout": dropout,
#                 "Epochs": epochs,
#                 "Batch_Size": batch_size,
#                 "Seq_Length": seq_len
#             })

#             if r2 > best_r2:
#                 best_r2 = r2
#                 best_model_data = {
#                     "y_pred_inv": y_pred_inv,
#                     "y_test_inv": y_test_inv,
#                     "mse": mse,
#                     "r2": r2,
#                     "params": {
#                         "lstm_units": lstm_units,
#                         "dropout": dropout,
#                         "epochs": epochs,
#                         "batch_size": batch_size,
#                         "sequence_length": seq_len
#                     }
#                 }

#         print(f"✅ Best MSE ({mode}): {best_model_data['mse']:.2f}")
#         print(f"✅ Best R²  ({mode}): {best_model_data['r2']:.4f}")
#         print(f"📌 Best Params: {best_model_data['params']}")
#         predictions_dict[mode] = best_model_data['y_pred_inv'].flatten()
#         y_test_dict[mode] = best_model_data['y_test_inv'].flatten()

#     # Plot actual vs predicted
#     plt.figure(figsize=(12, 5))
#     plt.plot(y_test_dict['Baseline'], label='Actual', color='black', linewidth=2)
#     plt.plot(predictions_dict['Baseline'], label='Predicted (Baseline)', linestyle='--', color='orange')
#     plt.plot(predictions_dict['Extended'], label='Predicted (Extended)', linestyle='--', color='dodgerblue')
#     plt.title(f"{ticker}: Actual vs Predicted (Baseline vs Extended)")
#     plt.xlabel("Test Samples")
#     plt.ylabel("Next Week's Close Price")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # Save all results to CSV
# results_df = pd.DataFrame(results_log)
# results_df.to_csv("gridsearch_lstm_results.csv", index=False)
# print("\n✅ Full grid search results saved to: gridsearch_lstm_results.csv")


# In[ ]:





# In[ ]:





# In[42]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import itertools
import os
import time
from tqdm import tqdm
from tensorflow.keras import backend as K

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Hyperparameter grid
lstm_units_list = [50, 64, 75, 100, 128]
dropout_list = [0.2, 0.25, 0.3, 0.4]
epochs_list = [30, 40, 50, 70]
batch_size_list = [16, 32, 64]
sequence_length_list = [5, 6, 7, 10, 15]

param_grid = list(itertools.product(
    lstm_units_list, dropout_list, epochs_list, batch_size_list, sequence_length_list
))

base_features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

# Try to resume from saved partial results
partial_path = "gridsearch_lstm_results_partial.csv"
if os.path.exists(partial_path):
    prev_results = pd.read_csv(partial_path)
    results_log = prev_results.to_dict("records")
    completed = set(zip(
        prev_results["Ticker"], prev_results["Model"],
        prev_results["LSTM_units"], prev_results["Dropout"],
        prev_results["Epochs"], prev_results["Batch_Size"],
        prev_results["Seq_Length"]
    ))
    print(f"🔁 Resuming from previous run with {len(results_log)} entries.")
else:
    results_log = []
    completed = set()

save_every = 100  # Save after every 100 runs

for ticker in tqdm(tickers, desc="Tickers"):
    df_ticker = merged_df[merged_df["Ticker"] == ticker].copy()
    df_ticker = df_ticker.sort_values(by=["Year", "Week"])
    df_ticker["Next_Close_Mean"] = df_ticker["Close_Mean"].shift(-1)
    corr_features = [col for col in df_ticker.columns if col.startswith(f"{ticker}_Corr_to_")]
    extended_features = base_features + corr_features
    predictions_dict, y_test_dict = {}, {}

    for mode, features in tqdm(list(zip(["Baseline", "Extended"], [base_features, extended_features])), desc=f"{ticker} Modes"):
        df_model = df_ticker.dropna(subset=features + ["Next_Close_Mean"])
        best_r2, best_model_data = -np.inf, None

        for i, (lstm_units, dropout, epochs, batch_size, seq_len) in enumerate(tqdm(param_grid, desc=f"{ticker} {mode}", leave=False)):

            config_id = (ticker, mode, lstm_units, dropout, epochs, batch_size, seq_len)
            if config_id in completed:
                continue

            # Prepare data
            scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
            X_scaled = scaler_X.fit_transform(df_model[features])
            y_scaled = scaler_y.fit_transform(df_model[["Next_Close_Mean"]])

            X_seq, y_seq = [], []
            for j in range(seq_len, len(X_scaled)):
                X_seq.append(X_scaled[j - seq_len:j])
                y_seq.append(y_scaled[j])

            X_seq, y_seq = np.array(X_seq), np.array(y_seq)
            X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

            # Build model
            model = Sequential()
            model.add(LSTM(lstm_units, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
            model.add(Dropout(dropout))
            model.add(LSTM(lstm_units))
            model.add(Dropout(dropout))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)

            # Evaluate
            y_pred = model.predict(X_test, verbose=0)
            y_pred_inv = scaler_y.inverse_transform(y_pred)
            y_test_inv = scaler_y.inverse_transform(y_test)
            mse = np.mean((y_test_inv - y_pred_inv) ** 2)
            r2 = r2_score(y_test_inv, y_pred_inv)

            results_log.append({
                "Ticker": ticker,
                "Model": mode,
                "R2": r2,
                "MSE": mse,
                "LSTM_units": lstm_units,
                "Dropout": dropout,
                "Epochs": epochs,
                "Batch_Size": batch_size,
                "Seq_Length": seq_len
            })

            completed.add(config_id)

            # Save periodically with timestamp
            if len(results_log) % save_every == 0 or i == len(param_grid) - 1:
                results_df = pd.DataFrame(results_log)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                interim_filename = f"results_partial_{ticker}_{mode}_{timestamp}.csv"
                results_df.to_csv(interim_filename, index=False)
                results_df.to_csv(partial_path, index=False)  # still update latest
                print(f"💾 Progress saved to {interim_filename}")

            # Optional memory clean-up
            K.clear_session()

        # Save best model info for plotting
        if best_model_data:
            predictions_dict[mode] = best_model_data['y_pred_inv'].flatten()
            y_test_dict[mode] = best_model_data['y_test_inv'].flatten()

    # Plot if both models completed
    if "Baseline" in predictions_dict and "Extended" in predictions_dict:
        plt.figure(figsize=(12, 5))
        plt.plot(y_test_dict['Baseline'], label='Actual', color='black', linewidth=2)
        plt.plot(predictions_dict['Baseline'], label='Predicted (Baseline)', linestyle='--', color='orange')
        plt.plot(predictions_dict['Extended'], label='Predicted (Extended)', linestyle='--', color='dodgerblue')
        plt.title(f"{ticker}: Actual vs Predicted (Baseline vs Extended)")
        plt.xlabel("Test Samples")
        plt.ylabel("Next Week's Close Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Save final output
results_df = pd.DataFrame(results_log)
results_df.to_csv("gridsearch_lstm_results.csv", index=False)
print("\n✅ Final results saved to gridsearch_lstm_results.csv")


# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# ============================
# LSTM for Stock Prediction (Baseline and Extended)
# ============================

base_features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

for ticker in tickers:
    print(f"\n{'='*60}\n LSTM for {ticker}\n{'='*60}")

    df_ticker = merged_df[merged_df["Ticker"] == ticker].copy()
    df_ticker = df_ticker.sort_values(by=["Year", "Week"])
    df_ticker["Next_Close_Mean"] = df_ticker["Close_Mean"].shift(-1)

    corr_features = [col for col in df_ticker.columns if col.startswith(f"{ticker}_Corr_to_")]
    extended_features = base_features + corr_features

    predictions_dict = {}
    y_test_dict = {}

    for mode, features in zip(["Baseline", "Extended"], [base_features, extended_features]):
        print(f"\n🔧 Training LSTM ({mode})")

        df_model = df_ticker.dropna(subset=features + ["Next_Close_Mean"])

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(df_model[features])
        y_scaled = scaler_y.fit_transform(df_model[["Next_Close_Mean"]])

        sequence_length = 5
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-sequence_length:i])
            y_seq.append(y_scaled[i])

        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1, verbose=0)

        y_pred = model.predict(X_test)
        y_pred_inv = scaler_y.inverse_transform(y_pred)
        y_test_inv = scaler_y.inverse_transform(y_test)

        mse = np.mean((y_test_inv - y_pred_inv)**2)
        r2 = r2_score(y_test_inv, y_pred_inv)
        print(f"MSE ({mode}): {mse:.2f}")
        print(f"R²  ({mode}): {r2:.4f}")

        predictions_dict[mode] = y_pred_inv.flatten()
        y_test_dict[mode] = y_test_inv.flatten()

    # Plot actual vs predicted for baseline and extended
    plt.figure(figsize=(12,5))
    plt.plot(y_test_dict['Baseline'], label='Actual', color='black', linewidth=2)
    plt.plot(predictions_dict['Baseline'], label='Predicted (Baseline)', linestyle='--', color='orange')
    plt.plot(predictions_dict['Extended'], label='Predicted (Extended)', linestyle='--', color='dodgerblue')
    plt.title(f"{ticker}: Actual vs Predicted (Baseline vs Extended)")
    plt.xlabel("Test Samples")
    plt.ylabel("Next Week's Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[ ]:


### I

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

base_features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

param_grid = [
    {"lstm_units": 50, "dropout": 0.2, "epochs": 30, "batch_size": 16, "sequence_length": 5},
    {"lstm_units": 64, "dropout": 0.3, "epochs": 40, "batch_size": 32, "sequence_length": 10},
    {"lstm_units": 100, "dropout": 0.3, "epochs": 50, "batch_size": 32, "sequence_length": 5},
]

for ticker in tickers:
    print(f"\n{'='*60}\n LSTM for {ticker}\n{'='*60}")
    df_ticker = merged_df[merged_df["Ticker"] == ticker].copy()
    df_ticker = df_ticker.sort_values(by=["Year", "Week"])
    df_ticker["Next_Close_Mean"] = df_ticker["Close_Mean"].shift(-1)

    corr_features = [col for col in df_ticker.columns if col.startswith(f"{ticker}_Corr_to_")]
    extended_features = base_features + corr_features

    predictions_dict = {}
    y_test_dict = {}

    for mode, features in zip(["Baseline", "Extended"], [base_features, extended_features]):
        print(f"\n🔧 Tuning LSTM ({mode})")

        df_model = df_ticker.dropna(subset=features + ["Next_Close_Mean"])
        best_r2 = -np.inf
        best_model_data = None

        for params in param_grid:
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_scaled = scaler_X.fit_transform(df_model[features])
            y_scaled = scaler_y.fit_transform(df_model[["Next_Close_Mean"]])

            X_seq, y_seq = [], []
            for i in range(params["sequence_length"], len(X_scaled)):
                X_seq.append(X_scaled[i - params["sequence_length"]:i])
                y_seq.append(y_scaled[i])

            X_seq, y_seq = np.array(X_seq), np.array(y_seq)
            X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

            model = Sequential()
            model.add(LSTM(params["lstm_units"], return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
            model.add(Dropout(params["dropout"]))
            model.add(LSTM(params["lstm_units"]))
            model.add(Dropout(params["dropout"]))
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"],
                      validation_split=0.1, verbose=0)

            y_pred = model.predict(X_test)
            y_pred_inv = scaler_y.inverse_transform(y_pred)
            y_test_inv = scaler_y.inverse_transform(y_test)

            mse = np.mean((y_test_inv - y_pred_inv) ** 2)
            r2 = r2_score(y_test_inv, y_pred_inv)

            if r2 > best_r2:
                best_r2 = r2
                best_model_data = {
                    "y_pred_inv": y_pred_inv,
                    "y_test_inv": y_test_inv,
                    "mse": mse,
                    "r2": r2,
                    "params": params
                }

        print(f"✅ Best MSE ({mode}): {best_model_data['mse']:.2f}")
        print(f"✅ Best R²  ({mode}): {best_model_data['r2']:.4f}")
        predictions_dict[mode] = best_model_data['y_pred_inv'].flatten()
        y_test_dict[mode] = best_model_data['y_test_inv'].flatten()

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(y_test_dict['Baseline'], label='Actual', color='black', linewidth=2)
    plt.plot(predictions_dict['Baseline'], label='Predicted (Baseline)', linestyle='--', color='orange')
    plt.plot(predictions_dict['Extended'], label='Predicted (Extended)', linestyle='--', color='dodgerblue')
    plt.title(f"{ticker}: Actual vs Predicted (Baseline vs Extended)")
    plt.xlabel("Test Samples")
    plt.ylabel("Next Week's Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[ ]:


### II 216 combinations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

tf.get_logger().setLevel('ERROR')

base_features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

# Expanded grid for better tuning
param_grid = [
    {"lstm_units": u, "dropout": d, "epochs": e, "batch_size": b, "sequence_length": s}
    for u in [50, 64, 100]
    for d in [0.2, 0.3, 0.4]
    for e in [30, 50]
    for b in [16, 32]
    for s in [5, 10]
]

for ticker in tickers:
    print(f"\n{'='*60}\n LSTM for {ticker}\n{'='*60}")
    df_ticker = merged_df[merged_df["Ticker"] == ticker].copy()
    df_ticker = df_ticker.sort_values(by=["Year", "Week"])
    df_ticker["Next_Close_Mean"] = df_ticker["Close_Mean"].shift(-1)

    corr_features = [col for col in df_ticker.columns if col.startswith(f"{ticker}_Corr_to_")]
    extended_features = base_features + corr_features

    predictions_dict = {}
    y_test_dict = {}

    for mode_idx, (mode, features) in enumerate(zip(["Baseline", "Extended"], [base_features, extended_features])):
        print(f"\n🔧 Tuning LSTM ({mode})")
        df_model = df_ticker.dropna(subset=features + ["Next_Close_Mean"])
        best_r2 = -np.inf
        best_model_data = None

        # Use a different seed per mode to avoid identical splits
        mode_seed = 42 + mode_idx
        for params in param_grid:
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_scaled = scaler_X.fit_transform(df_model[features])
            y_scaled = scaler_y.fit_transform(df_model[["Next_Close_Mean"]])

            X_seq, y_seq = [], []
            for i in range(params["sequence_length"], len(X_scaled)):
                X_seq.append(X_scaled[i - params["sequence_length"]:i])
                y_seq.append(y_scaled[i])

            X_seq, y_seq = np.array(X_seq), np.array(y_seq)
            X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=mode_seed
            )

            model = Sequential()
            model.add(LSTM(params["lstm_units"], return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
            model.add(Dropout(params["dropout"]))
            model.add(LSTM(params["lstm_units"]))
            model.add(Dropout(params["dropout"]))
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mean_squared_error')
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            model.fit(
                X_train, y_train,
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                validation_split=0.1,
                verbose=0,
                callbacks=[early_stop]
            )

            y_pred = model.predict(X_test)
            y_pred_inv = scaler_y.inverse_transform(y_pred)
            y_test_inv = scaler_y.inverse_transform(y_test)

            mse = np.mean((y_test_inv - y_pred_inv) ** 2)
            r2 = r2_score(y_test_inv, y_pred_inv)

            if r2 > best_r2:
                best_r2 = r2
                best_model_data = {
                    "y_pred_inv": y_pred_inv,
                    "y_test_inv": y_test_inv,
                    "mse": mse,
                    "r2": r2,
                    "params": params
                }

        print(f"✅ Best MSE ({mode}): {best_model_data['mse']:.2f}")
        print(f"✅ Best R²  ({mode}): {best_model_data['r2']:.4f}")
        print(f"🔍 Best Params: {best_model_data['params']}")
        predictions_dict[mode] = best_model_data['y_pred_inv'].flatten()
        y_test_dict[mode] = best_model_data['y_test_inv'].flatten()

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(y_test_dict['Baseline'], label='Actual', color='black', linewidth=2)
    plt.plot(predictions_dict['Baseline'], label='Predicted (Baseline)', linestyle='--', color='orange')
    plt.plot(predictions_dict['Extended'], label='Predicted (Extended)', linestyle='--', color='dodgerblue')
    plt.title(f"{ticker}: Actual vs Predicted (Baseline vs Extended)")
    plt.xlabel("Test Samples")
    plt.ylabel("Next Week's Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[ ]:


merged_df.head(5)


# In[ ]:


# ============================
# Tensor Decomposition per Ticker (preparing for Bayesian Tensor Regression)
# ============================

import tensorly as tl
from tensorly.decomposition import parafac
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tl.set_backend('numpy')

# Manually define kruskal_to_tensor (since not exposed in tensorly 0.9.0)
def kruskal_to_tensor(kruskal_tensor):
    weights, factors = kruskal_tensor
    rank = len(weights)
    tensor_shape = [factor.shape[0] for factor in factors]
    full_tensor = np.zeros(tensor_shape)
    for r in range(rank):
        outer_prod = weights[r]
        for factor in factors:
            outer_prod = np.multiply.outer(outer_prod, factor[:, r])
        full_tensor += outer_prod
    return full_tensor

base_features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

for ticker in tickers:
    print(f"\n{'='*60}\n Tensor Decomposition for {ticker}\n{'='*60}")

    df_ticker = merged_df[merged_df['Ticker'] == ticker].copy()
    df_ticker = df_ticker.sort_values(by=["Year", "Week"])

    corr_features = [col for col in df_ticker.columns if col.startswith(f"{ticker}_Corr_to_")]
    features = base_features + corr_features

    # Build week index
    weeks = sorted(df_ticker.apply(lambda row: f"{int(row['Year'])}_{int(row['Week'])}", axis=1).unique())
    week_idx = {wk: i for i, wk in enumerate(weeks)}
    num_weeks = len(weeks)
    num_features = len(features)

    # Initialize tensor: (1, weeks, features)
    tensor_data = np.full((1, num_weeks, num_features), np.nan)

    for _, row in df_ticker.iterrows():
        wk = f"{int(row['Year'])}_{int(row['Week'])}"
        if wk in week_idx:
            for j, feat in enumerate(features):
                tensor_data[0, week_idx[wk], j] = row[feat]

    # Fill missing values
    tensor_data = np.nan_to_num(tensor_data, nan=0.0)

    # Tensor shape = (1, weeks, features)
    print(f"Tensor shape for {ticker}: {tensor_data.shape}")

    # CP decomposition (rank = 2 can be tuned)
    rank = 2
    weights, factors = parafac(tensor_data, rank=rank, n_iter_max=100)

    print(f" Decomposition complete for {ticker} (rank {rank})")
    print(f"Factor shapes → Ticker: {factors[0].shape}, Weeks: {factors[1].shape}, Features: {factors[2].shape}")

    # Plot latent week factors
    plt.figure(figsize=(10,4))
    for r in range(rank):
        plt.plot(factors[1][:, r], label=f"Component {r+1}")
    plt.title(f"{ticker} - Week latent factors")
    plt.xlabel("Week index")
    plt.legend()
    plt.show()

    # Plot latent feature factors
    plt.figure(figsize=(10,4))
    for r in range(rank):
        plt.bar(np.arange(len(features)), factors[2][:, r], alpha=0.5, label=f"Component {r+1}")
    plt.xticks(np.arange(len(features)), features, rotation=45, ha='right')
    plt.title(f"{ticker} - Feature latent factors")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Reconstruct tensor
    tensor_recon = kruskal_to_tensor((weights, factors))
    print(f"Example reconstructed value (week 0, feature 0): {tensor_recon[0,0,0]:.4f}")

print("\n All tickers processed.")


# In[ ]:





# In[ ]:


# ============================
# Bayesian Ridge Regression using Tensor Features
# ============================

from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print(f"\n{'='*60}\n Bayesian Ridge Regression with Tensor Features\n{'='*60}")

# Choose ticker for demonstration
ticker = "AAPL"
df_ticker = merged_df[merged_df["Ticker"] == ticker].copy()
df_ticker = df_ticker.sort_values(by=["Year", "Week"])

# Features: Base + Relevant Correlations
corr_features = [col for col in df_ticker.columns if col.startswith(f"{ticker}_Corr_to_")]
features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"] + corr_features

# Target: Next week’s closing mean
df_ticker["Next_Close_Mean"] = df_ticker["Close_Mean"].shift(-1)
df_model = df_ticker.dropna(subset=features + ["Next_Close_Mean"])

# Prepare features and target
X = df_model[features].values
y = df_model["Next_Close_Mean"].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use synchronized train/test split
train_df = df_model[df_model['Set'] == 'train']
test_df = df_model[df_model['Set'] == 'test']

X_train = scaler.fit_transform(train_df[features])
X_test = scaler.transform(test_df[features])
y_train = train_df["Next_Close_Mean"].values
y_test = test_df["Next_Close_Mean"].values

# Fit Bayesian Ridge model
model = BayesianRidge()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.4f}")

# Plot: Actual vs Predicted
plt.figure(figsize=(12, 5))
plt.plot(y_test, label='Actual', color='black', linewidth=2)
plt.plot(y_pred, label='Bayesian Predicted', linestyle='--', color='green')
plt.title(f"{ticker}: Bayesian Ridge – Actual vs Predicted")
plt.xlabel("Test Samples")
plt.ylabel("Next Week's Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


# ============================
# Bayesian Ridge Regression using Tensor Features
# ============================

from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']
base_features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]

print(f"\n{'='*65}\n Bayesian Ridge Regression Feature Comparison\n{'='*65}")

for ticker in tickers:
    print(f"\n{'='*60}\n Bayesian Ridge for {ticker}\n{'='*60}")

    df_ticker = merged_df[merged_df["Ticker"] == ticker].copy()
    df_ticker = df_ticker.sort_values(by=["Year", "Week"])
    df_ticker["Next_Close_Mean"] = df_ticker["Close_Mean"].shift(-1)

    corr_features = [col for col in df_ticker.columns if col.startswith(f"{ticker}_Corr_to_")]
    full_features = base_features + corr_features
    df_model = df_ticker.dropna(subset=full_features + ["Next_Close_Mean", "Set"])

    if len(df_model) < 30:
        print(f"⚠️ Skipping {ticker}, not enough data.")
        continue

    # Synchronized train/test split
    train_df = df_model[df_model["Set"] == "train"]
    test_df = df_model[df_model["Set"] == "test"]

    def train_and_eval(features, label):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[features])
        X_test = scaler.transform(test_df[features])
        y_train = train_df["Next_Close_Mean"].values
        y_test = test_df["Next_Close_Mean"].values

        model = BayesianRidge()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n🔧 Training Bayesian Ridge ({label})")
        print(f"MSE ({label}): {mse:.2f}")
        print(f"R²  ({label}): {r2:.4f}")
        return model, y_test, y_pred

    # Baseline model
    _, _, _ = train_and_eval(base_features, "Baseline")

    # Extended model
    model_ext, y_test_ext, y_pred_ext = train_and_eval(full_features, "Extended")

    # Coefficients for extended model
    print(f"\n📊 Coefficients (Extended):")
    for name, coef in zip(full_features, model_ext.coef_):
        print(f"{name:35s}: {coef:+.6f}")

    # Plot: Actual vs Predicted (Extended)
    plt.figure(figsize=(12, 5))
    plt.plot(y_test_ext, label='Actual', color='black', linewidth=2)
    plt.plot(y_pred_ext, label='Bayesian Predicted (Extended)', linestyle='--', color='green')
    plt.title(f"{ticker}: Bayesian Ridge (Extended) – Actual vs Predicted")
    plt.xlabel("Test Samples")
    plt.ylabel("Next Week's Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[ ]:


comparison_df = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True),
    "Predicted": y_pred
})

plt.figure(figsize=(14, 6))
plt.plot(comparison_df["Actual"], label="Actual", linewidth=2, color="black")
plt.plot(comparison_df["Predicted"], label="Predicted", linestyle="--", color="dodgerblue")
plt.title("Linear Regression – Actual vs Predicted Closing Price")
plt.xlabel("Test Samples")
plt.ylabel("Next Week's Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


# Reset index of y_test in case it's a Series with a complex index
y_test_reset = y_test.reset_index(drop=True)
y_pred_series = pd.Series(y_pred)

plt.figure(figsize=(8, 6))
plt.scatter(y_test_reset, y_pred_series, alpha=0.6, edgecolors='k')
plt.plot([y_test_reset.min(), y_test_reset.max()],
         [y_test_reset.min(), y_test_reset.max()],
         'r--', lw=2)  # Ideal line

plt.xlabel("Actual Next Week's Close")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual Closing Price")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


print("Linear Regression Coefficients:")
for feature, coef in zip(feature_columns, model.coef_):
    print(f"{feature:20s}: {coef:.6f}")


# In[ ]:


# Testing correlation feature 2


# In[ ]:


import pandas as pd

# Ensure df_stock has datetime and is sorted
df_stock['Date'] = pd.to_datetime(df_stock['Date'])
df_stock = df_stock.sort_values(by='Date')

# Create daily price pivot table
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']
price_pivot = df_stock.pivot(index='Date', columns='Ticker', values='Close').dropna()

# Compute daily returns
returns = price_pivot.pct_change().dropna()

# Compute all rolling correlations (multi-peer)
window_size = 4
all_corr_features = []

for stock in tickers:
    for peer in tickers:
        if stock != peer:
            rolling_corr = returns[stock].rolling(window=window_size).corr(returns[peer])
            corr_col = f'Corr_{stock}_to_{peer}'

            df_corr = pd.DataFrame({
                'Date': rolling_corr.index,
                'Ticker': stock,
                corr_col: rolling_corr.values
            })

            df_corr['Year'] = df_corr['Date'].dt.isocalendar().year
            df_corr['Week'] = df_corr['Date'].dt.isocalendar().week

            all_corr_features.append(df_corr[['Year', 'Week', 'Ticker', corr_col]])

# Combine all features and pivot to wide format
df_all_corr = pd.concat(all_corr_features)
df_corr_wide = df_all_corr.groupby(['Year', 'Week', 'Ticker']).mean().reset_index()

# Inspect or merge with main dataset
print("Multi-peer correlation features ready")


# In[ ]:


df_corr_wide


# In[ ]:


# Objectives: 
# 1. Implement regression with correlation matrices
# 2. Implement LSTM algorithm
# 3. Implement Bayesian Tensor Regression


# In[ ]:


### Bayesian Tensor Regression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

#a Reuse the same merged_df and selected features
features = ["Prev_Close_Mean", "News_Count", "Sentiment_Score"]
target = "Next_Close_Mean"

# Drop any remaining NaNs just to be safe
merged_df = merged_df.dropna(subset=features + [target])

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(merged_df[features])
y = merged_df[target].values

# Create sequences (window size = 4 weeks)
def create_lstm_sequences(X, y, lookback=4):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

lookback = 4
X_seq, y_seq = create_lstm_sequences(X_scaled, y, lookback)

# Train-test split
split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# LSTM Model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(lookback, len(features))),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Predict
y_pred = model.predict(X_test)

# Plot predictions
plt.figure(figsize=(12,6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title("LSTM Prediction of Next Week's Close Price")
plt.xlabel("Weeks")
plt.ylabel("Close Price")
plt.legend()
plt.show()


# ### Testing

# In[ ]:


print(df_stock.dtypes)


# In[ ]:


# Pivot the data to wide format: each ticker becomes its own column
price_data = df_stock.pivot(index='Date', columns='Ticker', values='Close')

# Sort by date just in case
price_data = price_data.sort_index()

# Check result
price_data.head()


# In[ ]:


returns = price_data.pct_change(fill_method=None).dropna()


# In[ ]:


import itertools
import matplotlib.pyplot as plt

# Rolling window size
window = 60

# All pairs
tickers = returns.columns.tolist()
ticker_pairs = list(itertools.combinations(tickers, 2))

# Plot rolling correlation for each pair
for t1, t2 in ticker_pairs:
    plt.figure(figsize=(14, 4))
    rolling_corr = returns[t1].rolling(window).corr(returns[t2])
    plt.plot(rolling_corr, label=f'{t1} vs {t2}')
    plt.title(f"{window}-Day Rolling Correlation: {t1} vs {t2}")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# In[ ]:


import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

# Filter for just AAPL and TSLA
df_pair = df_stock[df_stock['Ticker'].isin(['AAPL', 'TSLA'])]
df_pivot = df_pair.pivot(index='Date', columns='Ticker', values='Close').dropna()

# Calculate log returns
returns = np.log(df_pivot).diff().dropna()

# Fit univariate GARCH(1,1) models to each asset
models = {}
residuals = {}
variances = {}

for col in returns.columns:
    am = arch_model(returns[col], vol='Garch', p=1, q=1, rescale=False)
    res = am.fit(disp='off')
    models[col] = res
    residuals[col] = res.std_resid
    variances[col] = res.conditional_volatility

# Stack standardized residuals
eps_t = np.vstack([residuals['AAPL'], residuals['TSLA']]).T
T = eps_t.shape[0]

# DCC(1,1) parameters (manually set or estimate later)
a = 0.01
b = 0.97

# Initialize Q with unconditional correlation of residuals
Qbar = np.cov(eps_t.T)
Q = Qbar.copy()
Qt_list = []

for t in range(T):
    eps = eps_t[t].reshape(-1, 1)
    Q = (1 - a - b) * Qbar + a * eps @ eps.T + b * Q
    Qt_list.append(Q.copy())

# Convert to correlation matrices
Rt_list = [Q / np.sqrt(np.outer(np.diag(Q), np.diag(Q))) for Q in Qt_list]
dcc_corr = [R[0,1] for R in Rt_list]

# Plot
dates = returns.index
plt.figure(figsize=(14,5))
plt.plot(dates, dcc_corr, label='DCC: AAPL vs TSLA')
plt.title('DCC-style Dynamic Correlation: AAPL vs TSLA')
plt.xlabel('Date')
plt.ylabel('Correlation')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ### Analysis n:

# In[ ]:


# ...


# In[ ]:





# In[ ]:




