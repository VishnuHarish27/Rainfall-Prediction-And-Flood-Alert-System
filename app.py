from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Read the data
data = pd.read_csv("rainfall in india 1901-2015.csv")

# Fill missing values with column means for numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Prepare data for Ridge Regression
# Grouping the data by 'SUBDIVISION' and selecting the required columns
group = data.groupby('SUBDIVISION')[['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']]
TN_data = group.get_group('TAMIL NADU')
df = TN_data.melt(id_vars=['YEAR']).reset_index(drop=True)
df.columns = ['Year', 'Month', 'Avg_Rainfall']
Month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
df['Month'] = df['Month'].map(Month_map)

# Split data into features (X) and target (y)
X = df[['Year', 'Month']]
y = df['Avg_Rainfall']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Define a pipeline for preprocessing and Random Forest Regression
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('rf', RandomForestRegressor())  # Random Forest Regression
])

# Hyperparameter tuning using GridSearchCV for Random Forest
param_grid_rf = {
    'rf__n_estimators': [100, 200, 300],  # Number of trees in the forest
    'rf__max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'rf__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'rf__min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)

# Train Random Forest Regression with best parameters
rf = grid_search_rf.best_estimator_
rf.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction_month = int(data['month'])
    prediction_year = int(data['year'])

    # Make predictions using the trained Random Forest Regression model
    prediction_data = pd.DataFrame({'Year': [prediction_year], 'Month': [prediction_month]})
    predicted_rainfall_cm = rf.predict(prediction_data)

    # Define threshold value for flood alert
    threshold_value = 100  # Define your threshold value here

    # Check if predicted rainfall exceeds threshold for flood alert
    if predicted_rainfall_cm[0] > threshold_value:
        alert_message = 'Flood Alert: Predicted rainfall exceeds threshold value!'
    else:
        alert_message = 'No flood alert: Predicted rainfall is within safe limits.'

    # Return the predicted rainfall along with the flood alert status as a JSON response
    return jsonify({
        'predicted_rainfall': f'{predicted_rainfall_cm[0]:.2f} cm',
        'flood_alert': alert_message
    })

if __name__ == '__main__':
    app.run(debug=True,port=8000)
