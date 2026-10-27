# Machine Learning - TP1: Data Exploration and Preprocessing

**Student:** Kaddour Abdelmalek Islam  
  

## Project Overview
This repository contains the solution for Practical Session 1 on data exploration and preprocessing using pandas, scikit-learn, and seaborn.

## Contents
- `TP1_Solutions.ipynb`: Jupyter notebook containing the code, outputs, and plots for all exercises.
- `data/`: Datasets used in the TP (`Iris_Dataset.csv`, `employees.csv`, `txt_file.txt`).
- `exercises/`: The lab instructions (`Practical session 1.pdf`).
- `requirements.txt`: Required python packages.

## Questions & Discussion Notes

### Exercise 1 - Data Splitting & Test Set
- **Input features:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`.  
- **Target:** `species`.
- **Copying the DataFrame:** Making a copy (`D.copy()`) allows us to experiment with cleaning and transformations without modifying the original data.
- **Why keep the test set separate?** The test set should only be used at the very end to evaluate generalization on unseen data. If we use test set information during preprocessing (like fitting scalers or imputers), it causes **data leakage** and gives overly optimistic results.

### Exercise 2 - Text Data
- The 20 Newsgroups training set has 11,314 documents and 20 target categories.
- Target values are integers (0 to 19) mapping to newsgroup topics in `target_names` (e.g., index 7 corresponds to `rec.autos`).

### Exercise 3 - Missing Values
- **Custom placeholders:** Pandas only recognizes standard null values (`NaN`, `None`) by default. Markers like `'-'` or `'na'` are treated as normal strings. We convert them to `np.nan` before handling them.
- **Imputation choice:**
  - `Salary` and `Bonus %`: Imputed using the median because it is more robust to outliers.
  - `Gender`, `Senior Management`, `Team`: Imputed using the most frequent value (mode).
  - Identifiers (`First Name`, `Start Date`, `Last Login Time`) are kept as they are since imputing personal identifiers doesn't make sense.

### Exercise 4 - Scaling and Encoding
- **Why scaling matters:** Distance-based models (like KNN or SVM) and gradient-based models are sensitive to differences in feature scales. Features with large values would dominate the distance calculation or slow down gradient descent.
- **Decision Trees:** Trees do not require feature scaling because splits are calculated on each feature independently based on thresholds.
- **One-Hot vs Ordinal Encoding:** `OrdinalEncoder` assigns integer codes (0, 1, 2, ...), which can mislead models into thinking there is a natural ordering. `OneHotEncoder` creates binary indicator columns, which is better for nominal categories with no order.

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook TP1_Solutions.ipynb
```
