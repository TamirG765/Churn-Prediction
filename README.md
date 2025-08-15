# Customer Churn Analysis & Prediction

I created this project to explore, analyze, and predict **customer churn** (whether a customer will leave or stay).  
It’s built as a single Jupyter Notebook: **`Churn_Analysis.ipynb`**, covering every step from **data cleaning** to **model evaluation**.

**Why this project is useful:**  
Predicting churn helps businesses take action before losing customers. My focus was on **catching as many churners as possible** (high recall) while keeping false alarms low.

---

## 📂 Dataset

- **File:** `CustomerChurn.csv` (Telco-style dataset)
- **Rows & Columns:** 7,043 rows × 21 columns
- **Target column:** `Churn` (Yes/No → converted to 1/0)
- **Main feature groups:**
  - **Demographics:** gender, senior citizen, partner, dependents
  - **Services:** phone service, internet service, security, backups, streaming, etc.
  - **Contract & Billing:** contract type, paperless billing, payment method
  - **Financial:** monthly charges, total charges
  - **Tenure:** how long the customer has been with the company

---

## 🗂 Project Structure

- **`Churn_Analysis.ipynb`** — main notebook with all analysis, preprocessing, and modeling.
- **`CustomerChurn.csv`** — dataset.

---

## 🔍 Data Exploration (EDA)

I explored the data to understand patterns and relationships:

- Checked for missing data and column types.
- Looked at each feature individually (bar charts for categories, histograms/KDE for numbers).
- Compared distributions of `MonthlyCharges` and `tenure` for churned vs. retained customers.
- Checked correlations (heatmap) to see which features have stronger relationships with churn.

**Insights:**
- Higher churn risk: month-to-month contracts, electronic check payments, no security/backup/tech support, fiber internet, short tenure.
- Lower churn risk: 2-year contracts, no internet service, long tenure (5+ years).
- Weak predictors: gender, phone service, multiple lines.

---

## ⚙️ Data Preprocessing

I built preprocessing **pipelines** to make the workflow clean and avoid data leakage:

- **Numeric columns**: scaled using `StandardScaler`.
- **Categorical columns**: one-hot encoded with `OneHotEncoder`.
- Dropped `customerID` and raw `tenure` (kept tenure bins instead).

**Class imbalance** (only ~26% churners) handled by:
- Using `class_weight='balanced'` in Logistic Regression and Random Forest.
- For XGBoost: used `scale_pos_weight` based on class counts.

---

## 🤖 Models

I used a **stratified 80/20 split** for train/test and trained three models:

1. **Logistic Regression** (scaled + encoded features)
2. **Random Forest**
3. **Gradient Boosting**  
   - Prefer XGBoost (if installed)  
   - Otherwise HistGradientBoostingClassifier

**Special step:**  
I **lowered the probability threshold to 0.30** (default is 0.50) to **catch more churners** — this increases recall for churn but may reduce precision.

**Optional:** Bayesian Optimization with `BayesSearchCV` to find better hyperparameters for each model.

---

## 📊 Results

Metrics: **AUC**, **Precision for No churn (class 0)**, **Recall for Churn (class 1)**

| Model              | AUC   | Precision (No Churn) | Recall (Churn) |
|--------------------|-------|----------------------|----------------|
| RandomForest       | 0.8348| 0.9343               | 0.8904         |
| LogisticRegression | 0.8294| 0.9551               | 0.9305         |
| XGBoost            | 0.8244| 0.9233               | 0.8583         |

**Takeaways:**
- Logistic Regression: highest recall for churners.
- Random Forest: best AUC, good all-around.
- XGBoost: close behind but slightly lower recall.

---
