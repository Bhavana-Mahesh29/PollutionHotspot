# 🌍 Urban Pollution Hotspot Detection & Green Infrastructure Recommendation

---

## 📌 Problem Statement

Urban areas are facing increasing levels of air pollution due to traffic, industrialization, and population growth. While pollution data is available, there is a lack of intelligent systems to identify high-risk pollution zones and recommend effective environmental solutions.

This project aims to:

* Detect pollution hotspots using Machine Learning
* Classify regions into pollution levels
* Recommend suitable green infrastructure solutions

---

## 🎯 Objective

* Classify areas into **Low, Medium, High pollution zones**
* Analyze pollution patterns using data
* Recommend green solutions like trees, green roofs, and urban forests

---

## ⚙️ Pipeline Diagram

```
Data Collection → Data Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction → Recommendation
```

---

## 📊 Dataset Details

* **Source:** https://data.opencity.in/organization/government-of-tamil-nadu
* **Format:** Excel (.xlsx)

### Features:

* PM2.5, PM10
* NO₂, SO₂, CO
* AQI Value
* Latitude & Longitude
* Date (Month extracted)

### Description:

The dataset contains pollution measurements across multiple locations and time periods. It is used to identify pollution patterns and classify areas based on pollution severity.

---

## 🔧 Data Preprocessing

* Converted pollutant values to numeric
* Handled missing values using mean imputation
* Extracted date features (month, day, weekday)
* Removed irrelevant and low-impact features

---

## 🧠 Feature Engineering

* Created **Pollution_Index** using weighted pollutants
* Generated target variable **Pollution_Zone**
* Applied outlier handling using IQR method

---

## 🤖 Model Details

### 🔹 Machine Learning Models:

* Logistic Regression
* Decision Tree
* Random Forest ⭐ (Best Performing)
* K-Nearest Neighbors (KNN)

### 🔹 Deep Learning Model:

* Feedforward Neural Network (TensorFlow/Keras)
* Dense layers with ReLU activation
* Dropout for regularization
* Softmax output for classification

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score (Primary Metric)

---

## 🌱 Recommendation System

Based on pollution levels:

| Pollution Level | Recommendation                                |
| --------------- | --------------------------------------------- |
| High            | Green Walls + Vertical Gardens / Urban Forest |
| Medium          | Roadside Trees + Green Roofs                  |
| Low             | Maintain Existing Greenery                    |

---

## 📊 Sample Output

* Classification of areas into:

  * Low
  * Medium
  * High pollution zones

* Example:

  * High → Green Walls
  * Medium → Roadside Trees
  * Low → Maintain Greenery

---

## 📁 Project Structure

```
pollution-hotspot-ml/
├── data/                          # Dataset
├── src/                           # Modular code (preprocessing, models)
├── main.py                        # Entry point
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
```

---

## 🚀 Steps to Run the Project

```bash
git clone https://github.com/Bhavana-Mahesh29/PollutionHotspot.git
cd PollutionHotspot
pip install -r requirements.txt
python main.py
```

---

## 📦 Required Libraries

* pandas
* numpy
* scikit-learn
* tensorflow
* matplotlib
* seaborn
* openpyxl
* joblib

---

## 👥 Team Members

* Aruna Shivani 24BCS030
* Bhavana M 24BCS045
* Dharshini G 24BCS065

---

## 🎯 Conclusion

This project demonstrates how Machine Learning and Deep Learning can be used to identify pollution hotspots and provide actionable environmental solutions. It supports data-driven decision-making for sustainable urban planning.

---
