# Rainfall-Prediction-And-Flood-Alert-System
# Introduction 
***  
Forecasting weather has always been a challenge for meteorologists worldwide due to its random nature. An Intelligent Rainfall Prediction and Flood Alert System using Machine Learning endeavours to revolutionize 
rainfall prediction and flood management by employing machine learning methods. The dataset contains historical rainfall data from 1901 to 2015 for various subdivisions in India. After filling missing values with column means, the data is visualized to understand annual and monthly rainfall trends. The analysis includes a specific focus on Tamil Nadu's rainfall patterns. For modelling, the data is prepared by grouping it by 'SUBDIVISION' and selecting relevant columns. A pipeline is created for preprocessing, including standardization, and Random Forest Regression is performed with hyperparameter tuning using GridSearchCV Hyperparameters, established prior to training, affect factors such as model complexity, learning rate, and regularization strength. Mean absolute error, Mean squared error, Root mean squared error, and R-squared are used to validate the model’s precision. The system allows for user input to predict rainfall for a given year and month, with a flood alert triggered if predicted rainfall exceeds a specified threshold.


