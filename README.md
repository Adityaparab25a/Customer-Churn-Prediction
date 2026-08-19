# Customer Churn Prediction System

An end-to-end Machine Learning application that predicts whether a customer is likely to churn based on their demographic, service, and account information.

The project uses **Logistic Regression** as the final prediction model and provides an interactive **Streamlit web application** where users can enter customer details and receive a churn probability and risk assessment.

---

## Project Overview

Customer churn occurs when a customer stops using a company's services.

The goal of this project is to build a machine learning system that can:

* Analyze customer information
* Predict the probability of customer churn
* Identify customers who are at higher risk of leaving
* Provide a user-friendly prediction interface
* Support customer retention decisions

Instead of simply predicting `0` or `1`, the application provides a **churn probability** and classifies the customer as having either **High Churn Risk** or **Low Churn Risk**.

---

## Key Features

* Data preprocessing and feature transformation
* Exploratory data analysis
* Multiple machine learning models
* Model performance comparison
* Accuracy, Precision, Recall and F1 evaluation
* ROC-AUC evaluation
* 5-fold cross-validation
* Classification threshold optimization
* Interactive Streamlit web application
* Churn probability prediction
* Customer risk classification
* Saved machine learning pipeline for deployment

---

## Machine Learning Workflow

```text
Customer Dataset
       ↓
Data Cleaning & Preprocessing
       ↓
Exploratory Data Analysis
       ↓
Train/Test Split
       ↓
Feature Encoding & Scaling
       ↓
Train Multiple ML Models
       ↓
Model Evaluation
       ↓
ROC-AUC & Cross-Validation
       ↓
Select Final Model
       ↓
Threshold Optimization
       ↓
Save Model Pipeline
       ↓
Streamlit Web Application
       ↓
Churn Prediction
```

---

## Models Evaluated

Five classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)

The models were compared using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* 5-fold Cross-Validation

---

## Model Comparison

### Test Set Performance

| Model                   |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| **Logistic Regression** | **80.38%** | **64.85%** |     57.22% | **60.80%** | **83.59%** |
| Decision Tree           |     78.96% |     60.21% | **61.50%** | **60.85%** |     82.96% |
| Random Forest           |     78.68% |     62.42% |     49.73% |     55.36% |     81.10% |
| KNN                     |     76.19% |     54.91% |     58.29% |     56.55% |     77.99% |
| SVM                     |     79.18% |     64.01% |     49.47% |     55.81% |     78.86% |

### 5-Fold Cross-Validation

| Model                   | Mean CV Accuracy | Mean CV Recall |
| ----------------------- | ---------------: | -------------: |
| **Logistic Regression** |       **80.23%** |     **54.72%** |
| Decision Tree           |           78.79% |         54.11% |
| Random Forest           |           78.58% |         49.23% |
| KNN                     |           76.46% |         53.91% |
| SVM                     |           80.07% |         49.30% |

---

## Final Model

**Logistic Regression** was selected as the final model.

It provided the strongest overall and most consistent performance across the evaluation methods.

The model achieved:

* Test Accuracy: **80.38%**
* Test Precision: **64.85%**
* Test Recall: **57.22%**
* Test F1 Score: **60.80%**
* ROC-AUC: **83.59%**
* Mean 5-Fold CV Accuracy: **80.23%**
* Mean 5-Fold CV Recall: **54.72%**

---

## Classification Threshold Optimization

The default classification threshold of `0.50` was evaluated against several alternative thresholds.

| Threshold |   Accuracy |  Precision |     Recall |   F1 Score |
| --------: | ---------: | ---------: | ---------: | ---------: |
|      0.30 |     74.27% |     51.08% | **75.94%** |     61.08% |
|      0.35 |     76.55% |     54.45% |     71.93% |     61.98% |
|  **0.40** | **78.39%** |     57.92% | **68.45%** | **62.75%** |
|      0.45 |     79.25% |     60.46% |     63.37% |     61.88% |
|      0.50 | **80.38%** | **64.85%** |     57.22% |     60.80% |
|      0.55 |     79.96% |     66.91% |     48.66% |     56.35% |
|      0.60 |     78.82% | **67.92%** |     38.50% |     49.15% |

A threshold of **0.40** was selected because it provided the highest F1 Score while substantially improving recall.

### Final Threshold Performance

At a threshold of `0.40`:

* Accuracy: **78.39%**
* Precision: **57.92%**
* Recall: **68.45%**
* F1 Score: **62.75%**

This allows the system to identify more customers who are actually likely to churn.

---

## Confusion Matrix at Threshold 0.40

```text
                 Predicted
              Stayed  Churned

Actually Stayed   847     186
Actually Churned  118     256
```

This means:

* **847** customers were correctly identified as staying.
* **186** customers were incorrectly flagged as churners.
* **118** actual churners were missed.
* **256** actual churners were correctly identified.

Compared with the original 0.50 threshold, the number of false negatives decreased from **160 to 118**, allowing the system to identify **42 additional churners**.

---

## Data Preprocessing

The project uses a preprocessing pipeline that handles both categorical and numerical features.

### Categorical Features

Categorical variables are transformed using:

```text
OneHotEncoder
```

### Numerical Features

Numerical variables are standardized using:

```text
StandardScaler
```

The preprocessing and Logistic Regression model are stored together in a single Scikit-learn Pipeline.

This ensures that new customer data receives the same preprocessing used during model training.

---

## Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Logistic Regression
* Decision Tree
* Random Forest
* KNN
* SVM

### Data Processing

* Pandas
* NumPy

### Visualization / Analysis

* Matplotlib
* Seaborn

### Web Application

* Streamlit

### Model Persistence

* Joblib

---

## Project Structure

```text
customer-churn-project/
│
├── app.py
├── customer_churn_model.pkl
├── requirements.txt
├── README.md
└── dataset.csv
```

### File Description

| File                       | Description                                          |
| -------------------------- | ---------------------------------------------------- |
| `app.py`                   | Streamlit web application                            |
| `customer_churn_model.pkl` | Trained preprocessing + Logistic Regression pipeline |
| `requirements.txt`         | Required Python packages                             |
| `README.md`                | Project documentation                                |
| `dataset.csv`              | Dataset used for model development                   |

---

## Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Navigate into the project:

```bash
cd customer-churn-project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

The user can enter customer information and click:

```text
Predict Churn
```

The system then displays:

* Churn probability
* Churn risk level
* Prediction based on the selected threshold

---

## Example Application Flow

```text
Enter Customer Details
          ↓
Click "Predict Churn"
          ↓
Model Calculates Probability
          ↓
Probability ≥ 0.40?
       ↙           ↘
     YES            NO
      ↓              ↓
High Churn Risk   Low Churn Risk
```

---

## Deployment

The application can be deployed using a cloud platform that supports Streamlit applications.

The deployment requires:

* `app.py`
* `customer_churn_model.pkl`
* `requirements.txt`

The dataset is not required for making predictions once the trained model has been saved.

---

## Limitations

The model's predictions are based on patterns present in the training dataset and should not be treated as guaranteed outcomes.

A prediction of high churn risk means the customer has characteristics associated with churn according to the trained model. It does not mean the customer will definitely leave.

Model performance may also change when the model is applied to data from a different population, company, or time period.

---

## Future Improvements

Potential future improvements include:

* Hyperparameter tuning
* Additional feature engineering
* Class imbalance handling
* More advanced ensemble models
* Model explainability using SHAP
* Customer retention recommendations
* Prediction history and analytics dashboard
* Database integration
* Automated model retraining
* Real-time prediction API
* Cloud deployment

---

## Conclusion

This project demonstrates an end-to-end Machine Learning workflow for customer churn prediction.

Multiple classification algorithms were trained and evaluated using standard classification metrics, ROC-AUC, and 5-fold cross-validation.

**Logistic Regression** was selected as the final model because it provided the strongest overall and most consistent performance.

The classification threshold was further optimized from `0.50` to `0.40`, increasing recall from **57.22% to 68.45%** and improving the F1 Score from **60.80% to 62.75%**.

The final model was integrated into a **Streamlit web application**, allowing users to enter customer information and receive an estimated churn probability and risk assessment.

---

## Author

**Aditya Parab**

Machine Learning / Data Science Project

---

## License

This project is intended for educational and demonstration purposes.
