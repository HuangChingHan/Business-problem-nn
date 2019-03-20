# -*- coding: utf-8 -*-
"""
Data : 2019/03/20

This is a classification of customer churn.

"""
# Step 1: Importing data
# Importing the libraries
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Step 2: Create matrix of features & target variable
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# Step 3: Encoding string variables into numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

# Step 4: Create dummy variable
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Step 5: Splitting the dataset into Training & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 6: Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 7: Importing required Modules
from keras.models import Sequential
from keras.layers import Dense

# Step 8: Define Network
# Initializing NN
classifier = Sequential()

# Add input layer & first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', input_dim=11, activation='relu'))

# Add second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# Add output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Step 9: Compile Network
classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 10: fitting our model
classifier.fit(X_train, y_train, batch_size=10, epochs=20)

# Step 11: Predicting test set
y_pred = classifier.predict(X_test)
# Conver probability into binary 0 & 1
y_pred = (y_pred > 0.5)

# Step 12: Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




