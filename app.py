from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the Flask app
app = Flask(__name__)

# Load the dataset (cancer.csv)
df = pd.read_csv('cancer.csv')

# Select the features
selected_features = ['Air Pollution', 'Genetic Risk', 'Obesity', 'Balanced Diet', 'OccuPational Hazards', 'Coughing of Blood']
X = df[selected_features]

# Target variable
y = df['Level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Predict on the test data and evaluate
y_pred = rf_clf.predict(X_test)

# Print the accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy on Test Data: ", accuracy*100)
print("Confusion Matrix:\n", conf_matrix)

# Route to serve the home page (HTML form)
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the user
    air_pollution = float(request.form['air_pollution'])
    genetic_risk = float(request.form['genetic_risk'])
    obesity = float(request.form['obesity'])
    balanced_diet = float(request.form['balanced_diet'])
    occupational_hazards = float(request.form['occupational_hazards'])
    coughing_of_blood = float(request.form['coughing_of_blood'])

    # Prepare the input data in the same format as the training data
    input_data = [[air_pollution, genetic_risk, obesity, balanced_diet, occupational_hazards, coughing_of_blood]]

    # Make a prediction
    prediction = rf_clf.predict(input_data)
    prediction_proba = rf_clf.predict_proba(input_data)

    # Get prediction class and probability
    predicted_class = prediction[0]
    predicted_prob = prediction_proba[0]

    # Return prediction results
    result = {
        'predicted_class': predicted_class,
        'predicted_probability': dict(zip(rf_clf.classes_, predicted_prob))
    }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
