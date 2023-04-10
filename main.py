import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import seaborn as sn
import warnings # current version generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
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

print("Question 2")
# Finding accuracy score of Naive Bayes on Titanic Data & correlation between some columns
titanic_data = pd.read_csv('Dataset/train.csv')  # Reads in data

# Part 1.2
data = titanic_data.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)  # Remove columns that get in the way of processing

data = data.dropna(axis=0)  # Drop rows with missing data

df = data.drop('Survived', axis=1)  # Separates the data we're analyzing
label = data['Survived']  # Saves column for doing spilt to find accuracy

df['Sex'] = df['Sex'].map({'male': True, 'female': False})  # Updates values to true or false so it can be processed

X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.2, random_state=0)  # Spilt training data

classifier = GaussianNB()  # Call GaussianNB constructor
classifier.fit(X_train, y_train)  # Fit data to model

y_pred = classifier.predict(X_test)  # Predict data from model

print("Correlation between 'Survived' & 'Sex' columns: ", data['Survived'].corr(df['Sex']))  # Print results of correlation
print("Data visualizations: ")
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
data['Sex'] = df['Sex'].map({'male': True, 'female': False})
sn.heatmap(data.corr(), annot=True)  # Only one heatmap shown
plt.show()

print('\nAccuracy is', accuracy_score(y_pred, y_test))  # Finds accuracy of model

# Question 3 - Glass Dataset
glass_dataset = pd.read_csv('Dataset/glass.csv')  # Reads in data

X = glass_dataset.iloc[:, :-1].values  # Separate values from file for training and testing
y = glass_dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)  # Spilt the data

print("GaussianNB Results:")
classifier_gaussian_nb = GaussianNB()  # Call constructor for GaussianNB
classifier_gaussian_nb.fit(X_train, y_train)  # Fit data to model
y_pred = classifier_gaussian_nb.predict(X_test)  # Find predictions of model

# Summary of the predictions made by the classifier
print("Classification Report: \n", classification_report(y_test, y_pred))
print('Accuracy is', accuracy_score(y_pred, y_test)) # Accuracy score

print("Data Visualization: ")  # Show data visualization
print("GaussianNB Confusion Matrix: \n", confusion_matrix(y_test, y_pred))  # Print confusion matrix

print("\nSVM Results: ")
classifier_svc = SVC()  # Do same as above with SVC
classifier_svc.fit(X_train, y_train)
y_pred = classifier_svc.predict(X_test)

# Summary of the predictions made by the classifier
print("Classification Report: \n", classification_report(y_test, y_pred))
print('Accuracy is', accuracy_score(y_pred, y_test)) # Accuracy score

print("Data Visualization: ")  # Show data visualization
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))  # Print Confusion matrix

sn.heatmap(glass_dataset.corr(), annot=True)  # Only one heatmap shown
plt.show()
