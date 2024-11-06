# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:14:08 2024

@author: SKV
"""

# Create a dual-axis chart comparing the average installs and revenue for 
# free vs. paid apps within the top 3 app categories. Apply filters to
# exclude apps with fewer than 10,000 installs and revenue below $10,000 
# and android version should be more than 4.0 as well as size should be 
# more than 15M and content rating should be Everyone and app name 
# should not have more than 30 characters including space and special character.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r'E:\NullClass\Play Store Data.csv')  
data.shape

#clean the installs and replace the , & + with space
data['Installs'] = data['Installs'].str.replace(',', '').str.replace('+', '')  
data['Installs'] = pd.to_numeric(data['Installs'], errors='coerce').fillna(0).astype(int)  

#clean th eprice and replace $ with space
data['Price'] = data['Price'].str.replace('$', '')  
data['Price'] = pd.to_numeric(data['Price'], errors='coerce').fillna(0) 

#clean the android ver columns
data['Android Ver'] = data['Android Ver'].astype(str).str.extract('(\d+\.\d+)')[0]  
data['Android Ver'] = pd.to_numeric(data['Android Ver'], errors='coerce')  

#clean the size

def clean_size(size):
    if 'M' in size:
        return float(size.replace('M', ''))
    elif 'k' in size:
        return float(size.replace('k', '')) / 1024
    return np.nan

data['Size'] = data['Size'].apply(clean_size)

#Apply filter for all the given cilumns 
filtered_data = data[
    (data['Installs'] >= 10000) & 
    (data['Size'] > 15) &  
    (data['Android Ver'] > 4.0) &  
    (data['Content Rating'] == 'Everyone') &  
    (data['App'].str.len() <= 30)  
    ]

len(filtered_data)
#1051

filtered_data['Revenue'] = filtered_data['Installs'] * filtered_data['Price']

filtered_data = filtered_data[filtered_data['Revenue'] > 10000]
len(filtered_data)
#33

# Group and aggregate data
average_data = filtered_data.groupby(['Category', 'Type']).agg({'Installs': 'mean', 'Revenue': 'mean'}).reset_index()

top_category = average_data.groupby('Category')agg({'Installs': 'mean', 'Revenue': 'mean'}).mean(axis=1).nlargest(3).index
top_data = average_data[average_data['Category'].isin(top_category)]
print(top_data)

# Add missing categories and types
for category in top_category:
    for app_type in ['Free', 'Paid']:
        if not ((top_data['Category'] == category) & (top_data['Type'] == app_type)).any():
            missing_data = pd.DataFrame({
                'Category': [category],
                'Type': [app_type],
                'Installs': [0],
                'Revenue': [0]
            })
            top_data = pd.concat([top_data, missing_data], ignore_index=True)

# Create dual-axis chart
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot average installs
ax1.bar(top_data['Category'] + " - " + top_data['Type'], top_data['Installs'], color='b', alpha=0.6, label='Average Installs')
ax1.set_xlabel('Category - Type')
ax1.set_ylabel('Average Installs', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis for revenue
ax2 = ax1.twinx()
ax2.plot(top_data['Category'] + " - " + top_data['Type'], top_data['Revenue'], color='r', marker='o', label='Average Revenue')
ax2.set_ylabel('Average Revenue ($)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add titles and legends
plt.title('Average Installs and Revenue for Free vs Paid Apps in Top 3 Categories')
fig.tight_layout()  
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show plot
plt.show()
