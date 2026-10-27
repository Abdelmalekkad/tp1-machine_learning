# Practical Session 1: Data Exploration and Preprocessing

**Student:** KADDOUR Abdelmalek Islam  
**Institution:** National Higher School of Advanced Technologies (ENSTA)  
**Email:** aa.kaddour@ensta.edu.dz  

## Overview
This repository contains the complete practical session deliverables for Machine Learning TP1, focusing on data exploration, train/test partitioning, text exploration, missing value detection & imputation, feature scaling, binarization, and categorical encoding.

---

## Technical Discussions

### 1. Test Set Isolation & Data Leakage (Exercise 1)
* **Why must the test set remain isolated until final evaluation?**
  The test set serves as a proxy for completely unseen real-world data. Its sole purpose is to provide an unbiased estimate of the model's true generalization performance.
* **Risks of test set contamination:**
  If test set observations are accessed during preprocessing (e.g., computing global scaling parameters, imputing missing values using the entire dataset, or selecting features), **data leakage** occurs. The model learns statistical properties of the evaluation set beforehand, producing unrealistically optimistic performance scores that fail to generalize to production data. All transformers must be fit *strictly* on the training split, and only applied via `transform()` to the test split.

### 2. Feature Scaling & Algorithm Sensitivity (Exercise 4A)
* **Why does feature scaling matter for distance-based and gradient-based algorithms?**
  * **Distance-based algorithms (k-NN, SVM with RBF kernel, k-Means):** These compute geometric distances (e.g., Euclidean distance). If one feature has a scale of $10^5$ and another has a scale of $10^0$, the feature with larger magnitudes dominates distance calculations, rendering smaller features effectively meaningless.
  * **Gradient-based models (Logistic Regression, Neural Networks):** Disparate feature scales lead to distorted loss surfaces (elongated ellipses), making gradient descent oscillate inefficiently. Scaling accelerates smooth and reliable convergence.
* **Is scaling needed for Decision Trees?**
  **No.** Decision trees and tree ensembles (Random Forests, Gradient Boosting) evaluate threshold splits on one feature independently at each node ($x_j \le \theta$). Because monotonic transformations do not change the sorting order of values within any single feature, decision trees are scale-invariant.

---

## Detailed Exercise Explanations

### Exercise 1 — Tabular Exploration and Splitting (`Iris_Dataset.csv`)
* **Inspection Outputs (Q2):**
  * `info()`: Reports column names, non-null value counts, and memory consumption.
  * `describe()`: Computes summary statistics (mean, std, min, quartiles, max) for numerical features.
  * `shape`: Returns dataset dimensions `(150, 5)` — 150 instances and 5 columns.
  * `columns`: Lists attribute names (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `species`).
  * `head()`: Displays the first 5 records for quick structural validation.
* **Utility of Data Copies (Q5):**
  Using `DataFrame.copy()` ensures that subsequent modifications, imputations, or experimental transforms do not mutate or corrupt the original raw dataset, preventing subtle side-effects.
* **Features vs. Target (Q7):**
  * **Input Features ($X$):** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`.
  * **Target ($y$):** `species` (categorical flower class).
  * `stratify=y` ensures that each class is represented in the exact same proportion (33.3% per class) across both the 80% train (120 rows) and 20% test (30 rows) subsets.

---

### Exercise 2 — Text Data Exploration
* **Dimensions & Labels (Q4):**
  The 20 Newsgroups training dataset contains **11,314 text documents** partitioned across **20 distinct newsgroup topics**.
* **Target Mapping (Q5 & Q6):**
  Integer labels index into the `target_names` array. For example, index `7` corresponds to category `'rec.autos'`. This numerical representation facilitates mathematical modeling while preserving the reference string classes.
* **Document Lengths (Q7):**
  Word counts exhibit significant variance (skewed distribution with short emails as well as very long discussion threads), highlighting the necessity of text normalization techniques (tokenization, stop-word removal, and TF-IDF weighting).

---

### Exercise 3 — Missing Value Detection & Imputation (`employees.csv`)
* **Why pandas ignores `'-'` and `'na'` (Q5):**
  Pandas identifies missingness primarily through standard null types (`NaN`, `None`). Arbitrary character strings such as `'-'`, `'na'`, `'N/A'`, or `'?'` are parsed as ordinary valid strings. Unless explicitly defined in `na_values` at read time or mapped later, `isna()` marks them as non-null.
* **Standardized Replacement Rule (Q6):**
  Values are converted to lowercase, stripped of surrounding whitespace, and tested against an agreed set of invalid markers `{"-", "na", "n/a", "missing", "?", ""}`. Matches are replaced with `np.nan`, followed by casting numeric columns with `pd.to_numeric`.
* **Imputation Strategy (Q7):**
  * **Numerical features (`Salary`, `Bonus %`):** Imputed with the `median` to provide robustness against extreme outliers.
  * **Categorical features (`Gender`, `Senior Management`, `Team`):** Imputed with the `most_frequent` (mode) strategy.
  * **Identifiers & Timestamps (`First Name`, `Start Date`, `Last Login Time`):** Kept intact; imputing individual identifiers creates synthetic identities without domain validity.
* **Limitations of Imputation (Q8):**
  Imputation approximates true underlying values. It can distort correlations, underestimate sample variance, and introduce bias if missingness is not completely at random (MCAR).

---

### Exercise 4 — Scaling, Binarization, and Categorical Encoding

#### Part A: Scalers Comparison
1. **`StandardScaler`:** Standardizes features by removing the mean and scaling to unit variance ($\mu = 0, \sigma = 1$). Feature-wise.
2. **`MinMaxScaler`:** Rescales features into a bounded interval $[0, 1]$ based on minimum and maximum values. Feature-wise.
3. **`MaxAbsScaler`:** Scales each feature by its maximum absolute value, mapping into $[-1, 1]$ while preserving sparsity and exact zeros. Feature-wise.
4. **`RobustScaler`:** Scales features using median and Interquartile Range ($IQR = Q_3 - Q_1$), making it resilient against severe outliers. Feature-wise.
5. **`normalize()`:** Rescales individual sample vectors (row-wise) to have unit norm ($L_2$ norm = 1). Sample-wise.
6. **`Binarizer(threshold)`:** Maps numerical values to binary output: $1$ if $x > \text{threshold}$, and $0$ otherwise.

#### Part B: Categorical Encoding
* **`OrdinalEncoder`:** Converts categorical attributes into numerical integers ($0, 1, 2, \dots$). The resulting integers represent arbitrary categorical codes rather than an inherent numerical order.
* **`LabelEncoder`:** Encodes 1D target labels into sequential integers with a clean mapping accessible via `classes_`.
* **`OneHotEncoder`:** Expands categorical variables into orthogonal binary indicator vectors ($K$ columns for $K$ unique levels).
* **Why One-Hot Encoding is preferred for nominal inputs:**
  Nominal features have no intrinsic ordering (e.g., countries, browser types). Assigning ordinal integers ($0, 1, 2, 3$) inadvertently implies a false numerical hierarchy ($3 > 0$) and misleading distance relations that linear and distance-based algorithms misinterpret. One-hot encoding treats all categories symmetrically and independently.

---

## Summary of Recommended Preprocessing Practices

| Data Type | Missing Value Strategy | Recommended Transformation |
| :--- | :--- | :--- |
| **Numerical** | Impute via `median` (robust to outliers) or `mean` | Scale with `StandardScaler` (Gaussian-like) or `RobustScaler` (with outliers) |
| **Categorical (Nominal)** | Impute with `most_frequent` (mode) | Encode with `OneHotEncoder` |
| **Categorical (Ordinal)** | Impute with `most_frequent` | Encode with `OrdinalEncoder` using explicit order mapping |
| **Text Data** | Remove null strings / empty documents | Tokenize, remove stopwords, and convert via CountVectorizer / TfidfVectorizer |
