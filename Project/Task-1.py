# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:31:04 2024

@author: SKV
"""

# Visualize the sentiment distribution (positive, neutral, negative) of 
# user reviews using a stacked bar chart, 
# segmented by rating groups (e.g., 1-2 stars, 3-4 stars, 4-5 stars). 
# Include only apps with more than 1,000 reviews and 
# group by the top 5 categories.

import pandas as pd

#Load the dataset 1. Playstoredata
play_data = pd.read_csv(r'E:\NullClass\Play Store Data.csv')

play_data.shape # (10841, 13)
play_data.columns
# ['App', 'Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type',
       # 'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver',
       # 'Android Ver']

#2. user_review

review = pd.read_csv(r'E:\NullClass\User Reviews.csv')

review.shape #(64295, 5)
review.columns
# ['App', 'Translated_Review', 'Sentiment', 'Sentiment_Polarity',
#        'Sentiment_Subjectivity']

#data cleaning
play_data.info()
play_data.isnull().sum()

#review
review.info()
review.isnull().sum()

play_data = play_data.dropna(subset = ['Rating'])

for i in play_data.columns:
    play_data[i].fillna(play_data[i].mode()[0], inplace= True)
    play_data.drop_duplicates(inplace=True)
    play_data = play_data[play_data['Rating'] <= 5]

review = review.dropna(subset=['Translated_Review'])
review.head() 

#In Installs to replace the , & + with space
play_data['Installs'] = play_data['Installs'].str.replace('+', '').str.replace(',', '').astype(int)
 
#for price to replace the $ with space(' ')
play_data['Price'] = play_data['Price'].str.replace('$', ' ').astype(float)

#Filter apps with more than 1000 reviews

play_data['Reviews'] = play_data['Reviews'].astype(int)

filtered_data = play_data[play_data['Reviews'] > 1000]
# 5485 rows x 13 columns

#identify top 5 categories
categories = filtered_data['Category'].value_counts().nlargest(5).index
filtered_data = filtered_data[filtered_data['Category'].isin(categories)]

#merge the datasets
merged_df = pd.merge(filtered_data, review, on='App', how= 'inner')
#[28984 rows x 17 columns]

# print(merged_df[merged_df['Category'] == 'FAMILY'].value_counts())

# Sentiment Analysis using VADER
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def classify_sentiment(review):
    score = sia.polarity_scores(review)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Netural'
    
# classify_sentiment('Looking forward app,')
# classify_sentiment('No recipe book Unable recipe book.')
# classify_sentiment('Excellent It really works')

#Apply sentiment analyser to review
merged_df['Sentiment_pred_label'] = merged_df['Translated_Review'].apply(classify_sentiment)

# merged_df['Sentiment_pred_label']=merged_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

merged_df['Sentiment_pred_label'].head()

#Define rating groups
def stars(rating):
    if rating <= 2:
        return '1-2 stars'
    elif rating <= 4:
        return '3-4 stars'
    else:
        return '4-5 stars'

stars(1)

merged_df['Stars'] = merged_df['Rating'].apply(stars)

# Group by Category and Sentiment
grouped_df = merged_df.groupby(['Category','Sentiment_pred_label']).size().unstack(fill_value=0)

#plot the stacked bar graph 

import matplotlib.pyplot as plt

ax = grouped_df.plot(kind='bar', stacked=True, figsize=(12, 8))

plt.title('Sentiment Distribution by Rating Groups and Categories')
plt.xlabel('Category')
plt.ylabel('Number of Reviews')
plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

for container in ax.containers:
    ax.bar_label(container, label_type='center')
    
plt.show()