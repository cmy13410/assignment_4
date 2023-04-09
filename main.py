import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Question 1 - Pandas
print("Question 1: \n")
df = pd.read_csv('data.csv')  # Reads csv file and saves to variable
print("data.csv file content: \n", df)  # Print results

data_desc = df.describe()  # Gets basic description about data
print("Description of Data: ")  # Print results
print(data_desc)

df = df.fillna(df.mean())  # For all the null values fills in mean
print(df.to_string())  # Prints results

aggregate_data = df.aggregate({"Duration":['min', 'max', 'count', 'mean'],  # Use of function to find min, max, count, mean of two columns
                                          "Maxpulse": ['min', 'max', 'count', 'mean']})
print("Aggregated data on Duration and Maxpulse columns: \n", aggregate_data)  # print results

filter_calories = df[(df['Calories'] > 500) & (df['Calories'] < 1000)]  # Filters based on range in Calories column
print("Filtered Calorie from 500-1000: \n", filter_calories)  # Print Results

filter_calories_pulse = df[(df['Calories'] > 500) & (df['Pulse'] < 100)]  # Filters based on range of Calories & Pulse columns
print("Filtered ranges on Calories and Pulse columns: \n", filter_calories_pulse)  # Print Results

df_modified = df[['Duration', 'Pulse', 'Calories']]  # Save specific columns to new dataframe
print("Copy of old data dataframe with no Maxpulse column\n", df_modified)   # Print Results

del df['Maxpulse']  # Delete Maxpulse from Dataframe
print("Updated DataFrame with 'Maxpulse' column now deleted:\n", df)   # Print Results

df['Calories'] = df['Calories'].astype(np.int64)  # Convert from float64 to int64
# Print Results
print("Updated Dataframe with 'Calories' column now having a int64 datatype:\n", df.dtypes)  # Print Results
print("Actual Dataframe: \n", df)

scatter_graph = df.plot.scatter(x='Duration', y='Calories', c='DarkBlue')
plt.show()

# Finding accuracy score of Naive Bayes on Titanic Data
# titanic_test = pd.read_csv('Dataset/train.csv')
