#Welcome to my project on crime in Chicago from 2001 to today! 
#Our main goal for the project is to analyse the crime data and to figure out if theres a link
#between the severity of the crime and the location of said crime. 

#We plan to clean up the dataset so its easily accessable for analysis work as well as 
#employ a machine learning algorithm to see which variables are mostly connected to
#an arrest. We are doing this to get a better grip on if either location or type of crime
#has a bigger impact on arrest rate. After that we will go deeper into the different versions
#of crimes and to see if they have a specific connection to locations. 

#As always, we start by importing the packages we are using
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler

#Loading the dataset and then checking if it got loaded in correctly
data = pd.read_csv('CrimesChicago.csv')

data.head()

data.isna().sum()

#As we are looking deeply into location, its safe for us to remove the empty rows
data = data.dropna(axis=0)

#Now lets check it again for null values and also see if theres any duplicates
data.isna().sum()

data.duplicated().sum()

#Checking the datatypes we can see that everything is in order
data.info()

data.columns

#Next up we will change the Date into two new columns called Month and Time of Day
data[['Month', 'Time']] = data['Date'].str.split('/', n=1, expand=True)
data[['Day', 'Time of Day']] = data['Time'].str.split(' ', n=1, expand=True)
data = data.drop(['Date','Time','Day','Time of Day'], axis=1)
data.head()

#Saving the cleaned dataset so we can use it for the dashboard
data.to_csv('clean_chicago',sep=',',index=False,encoding='utf-8')

#Since we are focusing on different types of crimes, lets check all the different types
data['Primary Type'].unique()
data['Description'].value_counts()
data['Primary Type'].value_counts()

#As we move forward we are going to be conducting a machine learning algorithm on the dataset
#and its important to drop any columns that might be in the way of a proper answer
data_mla = data.drop(['ID', 'Location Description', 'Case Number', 'IUCR', 'Block', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Updated On', 'Latitude', 'Longitude', 'Location','Month'], axis=1)

#As we can see, now we have the appropriate columns to conduct a proper machine learning algorithm 
data_mla.head()

#Next up we are one hot encoding categoricals
data_mla_dummy = pd.get_dummies(data_mla, columns=['Primary Type', 'Description'], drop_first=True)
data_mla_dummy.head()


#Because the dataset was so large, the kernel ended up crashing multiple times, 
#leavning me with no choice but to use a sample of 2 million. 
data_sample = data_mla_dummy.sample(n=2000000, random_state=9)

#Now that we have dummy encoded categoricals, we can go ahead and make test and training sets
#which we will use for the machine learning algorithm
y = data_sample['Arrest']
x = data_sample.copy()
x = x.drop('Arrest', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=42)

#We had a problem where the sample wouldnt predict any 1, either positive or negative,
#leaving us to use an undersampler, so that the distribution would be similar
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

#Next we are going to choose our parameters and instatiate our classifier
cv_params = {'max_depth':[4,5,6,7],
             'min_samples_leaf':[3,4,5],
             'min_samples_split':[2,3,4],
             'max_features':[3,4,5,6],
             'n_estimators':[75,100,125,150]}

rf = RandomForestClassifier(random_state=0)

scoring_ = ['precision','f1','recall','accuracy']

#We want to refit it to get the highest possible recall score,
#as this will give us a percentage of how many we were able to predict to be true,
#as we want to avoid the possibility of criminals going free
rf_cv = GridSearchCV(rf, cv_params, scoring=scoring_, cv=5, refit='recall')

rf_cv.fit(X_train_resampled, y_train_resampled)

#After fitting the machine learning program to the training sets, we check the best parameters and scores
rf_cv.best_params_

#As we can see the best score is approximately 0.75
rf_cv.best_score_

rf_cv.best_estimator_

#Next we make the prediction based off of X test, as we will use this is calculate the accuracy 
#as well as make the Confusion Matrix
y_pred = rf_cv.predict(X_test)

#The Recall is a 0.778
recall = recall_score(y_test, y_pred)
print(("Recall:", recall))
#The Accuracy for the predict is a solid 0.76
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
#The Precision is a 0.528
precision = precision_score(y_test, y_pred)
print("Precision:", precision)
#Lastly the F1 score is 0.629
f1 = f1_score(y_test, y_pred)
print("F1", f1)

#Plotting the confusion matrix we can see that the false negatives (Wrongly set free) are 28253, which is approximately between 1/4 and 1/3 
#of our True negative (Arrested), meaning that we are focusing on the right variable.
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

#Next we are plotting the feature importances to see which variables affect arrest rate the most
feature_importances = pd.Series(rf_cv.best_estimator_.feature_importances_, index= X_train.columns).sort_values(ascending=False)
#As we can see, Criminal Damage, Description "to property"
feature_importances.nlargest(20).plot.bar()
