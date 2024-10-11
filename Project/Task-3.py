# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:54:54 2024

@author: SKV
"""

# Create a violin plot to visualize the distribution of ratings for each app
# category, but only include categories with more than 50 apps and 
# app name should contain letter “C” and exclude apps with fewer than 10 reviews
# and rating should be less 4.0. This graph should not work between 6 PM to 11PM.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

play_store = pd.read_csv(r'E:\NullClass\Play Store Data.csv')  
play_store.shape

#get the current time
current_time = datetime.now().time()

#check the curret time not between 6PM to 11PM 

start_time = datetime.strptime("06:00 PM", "%I:%M %p").time()
end_time = datetime.strptime("11:00 PM", "%I:%M %p").time()

if not (start_time <= current_time <= end_time):
  
    # count apps in each category & filter category in more than 50 apps
        count_category = play_store['Category'].value_counts()
        valid_category = count_category[count_category > 50].index
        filtered_app = play_store[play_store['Category'].isin(valid_category)]

#filtered the merged dataframe

        filtered_app = filtered_app[
            (filtered_app['App'].str.contains('C', case=False)) &
            (filtered_app['Rating'] < 4.0) &
            (filtered_app['Reviews'] < 10)
            ]
    #   len(filtered_app)
    
#plot the violin graph

        plt.figure(figsize=(15,6))
        sns.violinplot(data=filtered_app,x='Category', y='Rating')
        plt.xlabel('Category')
        plt.ylabel('Rating')
        plt.title('Distribution of Rating for app category')
        plt.xticks(rotation=45, ha='right')
        plt.show()
else:
    print('The graph cannot be generated by 06:00PM to 11:00PM')