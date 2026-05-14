import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignore warning messages to keep the terminal clean
warnings.filterwarnings('ignore')

print("Loading data...")
# 1. Load Data
df = pd.read_csv("corona_tested_006 (2).csv")

print("Preprocessing data...")
# 2. Preprocess Data
# We only want to train on definite 'positive' or 'negative' cases
df = df[df['Corona'].isin(['positive', 'negative'])].copy()
df['Corona'] = df['Corona'].map({'positive': 1, 'negative': 0})

# Map symptoms to 1 (True) and 0 (False)
symptom_cols = ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache']
for col in symptom_cols:
    df[col] = df[col].astype(str).str.lower().map({'true': 1, 'false': 0}).fillna(0).astype(int)

# Map Age (Yes = 1, No = 0)
df['Age_60_above'] = df['Age_60_above'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)

# Map Sex (Male = 1, Female = 0)
df['Sex'] = df['Sex'].astype(str).str.lower().map({'male': 1, 'female': 0}).fillna(0).astype(int)

# Map Known Contact severity to numbers (Confirmed contact=2, Abroad=1, Other=0)
contact_map = {'contact with confirmed': 2, 'abroad': 1, 'other': 0}
df['Known_contact'] = df['Known_contact'].astype(str).str.lower().map(contact_map).fillna(0).astype(int)

# Define Features (X) and Target (y)
features = symptom_cols + ['Age_60_above', 'Sex', 'Known_contact']
X = df[features]
y = df['Corona']

# 3. Split Data
print("Splitting data...")
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

# 4. Correlation Analysis (Optional terminal output)
print("\n--- Feature Correlation with Corona ---")
data_for_corr = X_train.copy()
data_for_corr['Corona'] = y_train
correlation_matrix = data_for_corr.corr()
print(correlation_matrix['Corona'].sort_values(ascending=False).to_string())
print("---------------------------------------\n")

# 5. Train & Evaluate the Model
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

print("Evaluating the model...")
val_preds = model.predict(X_val)
test_preds = model.predict(X_test)
print(f"Validation Accuracy: {accuracy_score(y_val, val_preds)*100:.2f}%")
print(f"Test Accuracy:       {accuracy_score(y_test, test_preds)*100:.2f}%")

# 6. Create the Chatbot / Dashboard
print("\nLaunching the interactive dashboard...")

def predict_corona(cough, fever, sore_throat, shortness_of_breath, headache, age_60, sex, contact):
    
    # Format inputs into a pandas DataFrame exactly like the training data
    input_data = pd.DataFrame({
        'Cough_symptoms': [1 if cough else 0],
        'Fever': [1 if fever else 0],
        'Sore_throat': [1 if sore_throat else 0],
        'Shortness_of_breath': [1 if shortness_of_breath else 0],
        'Headache': [1 if headache else 0],
        'Age_60_above': [1 if age_60 else 0],
        'Sex': [1 if sex == 'Male' else 0],
        'Known_contact': [2 if contact == 'Contact with confirmed' else (1 if contact == 'Abroad' else 0)]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100  # Get % chance of being positive
    
    if prediction == 1:
        return f"⚠️ High Likelihood of COVID-19 ({probability:.1f}% AI Confidence). Please consult a healthcare professional and self-isolate."
    else:
        return f"✅ Low Likelihood of COVID-19 ({probability:.1f}% AI Confidence). However, continue monitoring your symptoms."

# Define the user interface
inputs = [
    gr.Checkbox(label="Do you have a Cough?"),
    gr.Checkbox(label="Do you have a Fever?"),
    gr.Checkbox(label="Do you have a Sore Throat?"),
    gr.Checkbox(label="Do you experience Shortness of Breath?"),
    gr.Checkbox(label="Do you have a Headache?"),
    gr.Checkbox(label="Are you 60 years or older?"),
    gr.Radio(choices=["Male", "Female", "Other"], label="Sex", value="Other"),
    gr.Dropdown(choices=["Other", "Abroad", "Contact with confirmed"], label="Recent Known Contact", value="Other")
]

# Create the dashboard
app = gr.Interface(
    fn=predict_corona, 
    inputs=inputs, 
    outputs=gr.Textbox(label="AI Diagnosis Prediction"), 
    title="🩺 COVID-19 Symptom Checker Bot",
    description="Check your symptoms below. This AI was trained on testing data to evaluate the statistical likelihood of COVID-19 based on symptoms and exposure history."
)

# Launch the dashboard
# Removed 'inline=True' because we are running this as a normal Python script now
app.launch()