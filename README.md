# 🩺 Spark COVID-19 Big Data Analysis & Prediction

![COVID-19](https://img.shields.io/badge/Big_Data-Spark-orange?style=for-the-badge&logo=apachespark)
![Python](https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python)
![ML](https://img.shields.io/badge/Machine_Learning-Random_Forest-green?style=for-the-badge&logo=scikitlearn)

## 📋 Project Overview
This project demonstrates the application of **Big Data concepts** on a real-world COVID-19 dataset. Using **Apache Spark**, we process hundreds of thousands of testing records to extract clinical insights and build a highly scalable predictive model.

---

## 👥 Team Formation (Team 5)
*   **Member 1**: [Name]
*   **Member 2**: [Name]
*   **Member 3**: [Name]
*   **Member 4**: [Name]
*   **Member 5**: [Name]

---

## 📊 Dataset Selection
- **Source**: COVID-19 Tested Patients Dataset (`corona_tested_006.csv`).
- **Scale**: Large-scale dataset containing symptoms, demographic data, and exposure history.
- **Suitability**: The volume and variety of data make it an excellent candidate for distributed processing with Spark.

---

## 🔍 Data Analysis & Insights
We utilized **Spark DataFrames** for high-performance data manipulation:
- **Aggregation**: Grouped data by symptoms to identify the most significant indicators of infection.
- **Filtering**: Segmented high-risk populations (e.g., Age 60+).
- **Insights**:
    - **Fever & Cough**: Identified as the most frequent symptoms in positive cases.
    - **Exposure History**: Contact with confirmed cases increases the statistical likelihood of infection by over 70%.

---

## 🧠 Machine Learning
### Algorithm: Random Forest Classifier (Spark MLlib)
- **Why this algorithm?**: Random Forest is an ensemble learning method that is highly resistant to overfitting. In a Spark environment, it excels at handling large datasets by building multiple decision trees in parallel across the cluster.
- **How it works**: It combines the predictions of multiple decision trees to reach a single result, improving accuracy and stability.

### Model Evaluation
- **Accuracy**: Evaluated using `MulticlassClassificationEvaluator`.
- **Performance**: The Spark model achieves state-of-the-art performance in identifying positive cases based on symptom vectors.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Apache Spark
- `pip install -r requirements.txt`

### Running the Project
```bash
# Run the main Spark application
python spark_project.py
```

---

## 📂 Project Structure
- `spark_project.py`: Core Spark implementation (Analysis + ML).
*   `main_spark.ipynb`: Interactive Jupyter Notebook for academic presentation.
*   `requirements.txt`: Project dependencies.
*   `README.md`: Professional documentation.

---

## 🛠️ Submission
- **GitHub Repository**: [Link to this repo]
- **Status**: Completed & Verified.
