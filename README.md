# Cancer Prediction with randomforest and web interface 🧬  

This project is a **Flask-based web application** that predicts cancer risk levels using a **Random Forest Classifier** trained on medical and lifestyle features.  

---

## 🚀 Features
- Web-based UI with Flask  
- Takes user inputs such as:
  - Air Pollution  
  - Genetic Risk  
  - Obesity  
  - Balanced Diet  
  - Occupational Hazards  
  - Coughing of Blood  
- Returns predicted cancer risk level along with class probabilities  
- Displays model evaluation metrics in console (Accuracy, Confusion Matrix, Classification Report)  

---

## 📂 Project Structure

- ├── cancer.csv # Dataset file
- ├── app.py # Main Flask application
- ├── templates/
- │ └── index.html # Frontend template
- ├── static/ # (Optional) for CSS/JS files
- └── README.md # Project documentation

## Run the Flask app
python app.py

## Model Details

Algorithm: Random Forest Classifier

Evaluation Metrics Printed in Console:

- Accuracy 99%

- Confusion Matrix 
 [[119   0   0]
 [  0  84   0]
 [  0   0  97]]



