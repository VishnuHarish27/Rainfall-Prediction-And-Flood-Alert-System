# Rainfall-Prediction-And-Flood-Alert-System
# Introduction 
Forecasting weather has always been a challenge for meteorologists worldwide due to its random nature. An Intelligent Rainfall Prediction and Flood Alert System using Machine Learning endeavours to revolutionize 
rainfall prediction and flood management by employing machine learning methods. The dataset contains historical rainfall data from 1901 to 2015 for various subdivisions in India. After filling missing values with column means, the data is visualized to understand annual and monthly rainfall trends. The analysis includes a specific focus on Tamil Nadu's rainfall patterns. For modelling, the data is prepared by grouping it by 'SUBDIVISION' and selecting relevant columns. A pipeline is created for preprocessing, including standardization, and Random Forest Regression is performed with hyperparameter tuning using GridSearchCV Hyperparameters, established prior to training, affect factors such as model complexity, learning rate, and regularization strength. Mean absolute error, Mean squared error, Root mean squared error, and R-squared are used to validate the model’s precision. The system allows for user input to predict rainfall for a given year and month, with a flood alert triggered if predicted rainfall exceeds a specified threshold.
# Architecture 
![image](https://github.com/VishnuHarish27/Rainfall-Prediction-And-Flood-Alert-System/assets/138471302/f61f7273-cc05-4fbc-b4f6-78f3d640752c)
# Requirements
Python>=3.8
flask
numpy
pandas
sklearn
skicit-learn
# How to run :
Step 1 : "rainfall in india 1901-2015" file contains the historical dataset.<br>
Step 2 : Templates folder contains the html which are used in the flask application.<br>
Step 3 : Static folder contains the css file which is used for the html and also contains images and audio file.<br>
Step 4 : Run the app.py file.<br>
Step 5 : It will give the IP which is to be pasted in the browser renders our flask application.<br>
# Output
![image](https://github.com/VishnuHarish27/Rainfall-Prediction-And-Flood-Alert-System/assets/138471302/e0c520c7-10d5-4ce8-8aa8-d11d314bc4c1 "title-1") 

