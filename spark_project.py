import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower, count, avg
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import gradio as gr
import pandas as pd

# Initialize Spark Session
print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("Covid19BigDataAnalysis") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Set Log Level to avoid too much noise
spark.sparkContext.setLogLevel("ERROR")

def load_and_preprocess_data():
    print("Loading data with Spark...")
    # Load dataset
    file_path = "corona_tested_006 (2).csv"
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    print("Preprocessing data using Spark DataFrames...")
    
    # 1. Filter for only positive/negative cases
    df = df.filter(col("Corona").isin(["positive", "negative"]))
    
    # 2. Map Target: positive -> 1, negative -> 0
    df = df.withColumn("label", when(col("Corona") == "positive", 1).otherwise(0))
    
    # 3. Clean and Map Symptoms (Cough_symptoms, Fever, Sore_throat, Shortness_of_breath, Headache)
    symptom_cols = ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache']
    for c in symptom_cols:
        df = df.withColumn(c, when(lower(col(c).cast("string")) == "true", 1).otherwise(0))
    
    # 4. Map Age (Yes -> 1, No -> 0)
    df = df.withColumn("Age_60_above", when(lower(col("Age_60_above").cast("string")) == "yes", 1).otherwise(0))
    
    # 5. Map Sex (Male -> 1, Female -> 0)
    df = df.withColumn("Sex", when(lower(col("Sex").cast("string")) == "male", 1).otherwise(0))
    
    # 6. Map Known Contact (Confirmed contact=2, Abroad=1, Other=0)
    df = df.withColumn("Known_contact", 
                       when(lower(col("Known_contact").cast("string")) == "contact with confirmed", 2)
                       .when(lower(col("Known_contact").cast("string")) == "abroad", 1)
                       .otherwise(0))
    
    return df

def perform_analysis(df):
    print("\n" + "="*50)
    print("BIG DATA ANALYSIS & INSIGHTS (Spark DataFrames)")
    print("="*50)
    
    # Total Count
    total_records = df.count()
    print(f"Total processed records: {total_records}")
    
    # Insights: Positive vs Negative distribution
    df.groupBy("Corona").count().show()
    
    # Insight: Symptom prevalence in Positive cases
    print("Prevalence of symptoms in Positive cases:")
    positive_df = df.filter(col("label") == 1)
    positive_df.select([avg(col(c)).alias(f"Avg_{c}") for c in ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache']]).show()
    
    # Insight: Age group risk
    print("Risk analysis by Age (60+):")
    df.groupBy("Age_60_above", "Corona").count().show()
    
    # Insight: Contact history impact
    print("Impact of Known Contact:")
    df.groupBy("Known_contact", "Corona").count().show()
    print("="*50 + "\n")

def train_spark_ml(df):
    print("Preparing data for Machine Learning...")
    
    # Define feature columns
    feature_cols = ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache', 'Age_60_above', 'Sex', 'Known_contact']
    
    # VectorAssembler to combine features into a single vector column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(df).select("features", "label")
    
    # Split data (80% train, 20% test)
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    
    print("Training Random Forest Classifier on Spark...")
    # Why Random Forest? It handles categorical features well and is less prone to overfitting than a single decision tree.
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=10)
    model = rf.fit(train_data)
    
    print("Evaluating model...")
    predictions = model.transform(test_data)
    
    # Evaluation
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    return model, assembler

def launch_interface(model, assembler):
    print("Launching Gradio Dashboard...")
    
    def predict_corona(cough, fever, sore_throat, shortness_of_breath, headache, age_60, sex, contact):
        # Prepare input data for Spark
        sex_val = 1 if sex == 'Male' else 0
        contact_val = 2 if contact == 'Contact with confirmed' else (1 if contact == 'Abroad' else 0)
        
        # In a real Spark production environment, we might use Spark Streaming or a Model Server.
        # For this local demo, we'll create a single-row DataFrame.
        input_dict = {
            'Cough_symptoms': 1 if cough else 0,
            'Fever': 1 if fever else 0,
            'Sore_throat': 1 if sore_throat else 0,
            'Shortness_of_breath': 1 if shortness_of_breath else 0,
            'Headache': 1 if headache else 0,
            'Age_60_above': 1 if age_60 else 0,
            'Sex': sex_val,
            'Known_contact': contact_val
        }
        
        # Convert to Spark DataFrame
        input_df = spark.createDataFrame([input_dict])
        
        # Transform using assembler
        input_data = assembler.transform(input_df)
        
        # Make prediction
        prediction_df = model.transform(input_data)
        result = prediction_df.collect()[0]
        
        prediction = result['prediction']
        probability = result['probability'][1] * 100 # Probability of class 1 (Positive)
        
        if prediction == 1.0:
            return f"⚠️ High Likelihood of COVID-19 ({probability:.1f}% AI Confidence). Please consult a healthcare professional."
        else:
            return f"✅ Low Likelihood of COVID-19 ({probability:.1f}% AI Confidence). Continue monitoring your health."

    inputs = [
        gr.Checkbox(label="Cough?"),
        gr.Checkbox(label="Fever?"),
        gr.Checkbox(label="Sore Throat?"),
        gr.Checkbox(label="Shortness of Breath?"),
        gr.Checkbox(label="Headache?"),
        gr.Checkbox(label="60 years or older?"),
        gr.Radio(choices=["Male", "Female"], label="Sex", value="Female"),
        gr.Dropdown(choices=["Other", "Abroad", "Contact with confirmed"], label="Recent Known Contact", value="Other")
    ]

    app = gr.Interface(
        fn=predict_corona, 
        inputs=inputs, 
        outputs=gr.Textbox(label="Spark AI Prediction"), 
        title="🩺 Big Data COVID-19 Predictor (Powered by Spark)",
        description="This predictor uses a Spark ML Random Forest model trained on large-scale testing data."
    )
    
    app.launch()

if __name__ == "__main__":
    df = load_and_preprocess_data()
    perform_analysis(df)
    model, assembler = train_spark_ml(df)
    launch_interface(model, assembler)
