import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD AND PREPROCESS THE ORIGINAL DATA
# ==========================================
print("Loading and preprocessing original data...")
df = pd.read_csv("corona_tested_006 (2).csv")

# Filter positive/negative
df = df[df['Corona'].isin(['positive', 'negative'])].copy()
df['Corona'] = df['Corona'].map({'positive': 1, 'negative': 0})

# Map symptoms and other features
symptom_cols = ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache']
for col in symptom_cols:
    df[col] = df[col].astype(str).str.lower().map({'true': 1, 'false': 0}).fillna(0).astype(int)

df['Age_60_above'] = df['Age_60_above'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)
df['Sex'] = df['Sex'].astype(str).str.lower().map({'male': 1, 'female': 0}).fillna(0).astype(int)
contact_map = {'contact with confirmed': 2, 'abroad': 1, 'other': 0}
df['Known_contact'] = df['Known_contact'].astype(str).str.lower().map(contact_map).fillna(0).astype(int)

# Keep only the features we need + the target
features = symptom_cols + ['Age_60_above', 'Sex', 'Known_contact']
df = df[features + ['Corona']]

# ==========================================
# 2. SPLIT DATA INTO TWO FILES & SAVE
# ==========================================
print("Splitting data into two separate files...")
# 80% for Training & Validation, 20% for Final Testing
df_train_val, df_final_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Corona'])

# Save to physical CSV files
df_train_val.to_csv("train_val_data.csv", index=False)
df_final_test.to_csv("final_test_data.csv", index=False)
print("✅ Saved 'train_val_data.csv'")
print("✅ Saved 'final_test_data.csv'")

# ==========================================
# 3. LOAD FILE 1 AND TRAIN/VALIDATE (WITH EPOCHS)
# ==========================================
print("\nLoading File 1 (Train & Validate) for Neural Network...")
train_val_data = pd.read_csv("train_val_data.csv")

X_train_val = train_val_data[features]
y_train_val = train_val_data['Corona']

# Split the loaded data into Training (80%) and Validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Build a Deep Learning Neural Network
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    Dense(8, activation='relu'),                                    # Hidden layer
    Dense(1, activation='sigmoid')                                  # Output layer (0 to 1 probability)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n🚀 Starting Training Process over 20 Epochs...")
# Train the model and save the history
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=20,          # Number of passes through the data
    batch_size=32       # How many rows to process at once
)

# ==========================================
# 4. PLOT THE TRAINING VS VALIDATION ACCURACY
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig("epoch_accuracy_plot.png")
print("\n✅ Saved accuracy graph as 'epoch_accuracy_plot.png'")

# ==========================================
# 5. LOAD FILE 2 AND PERFORM FINAL TEST
# ==========================================
print("\nLoading File 2 (Final Test Data)...")
test_data = pd.read_csv("final_test_data.csv")

X_final_test = test_data[features]
y_final_test = test_data['Corona']

print("Evaluating on Final Unseen Test Data...")
test_loss, test_accuracy = model.evaluate(X_final_test, y_final_test)

print("\n" + "="*40)
print(f" FINAL TEST ACCURACY: {test_accuracy * 100:.2f}%")
print("="*40)