4\. Experimental Setup
======================

4.1 Dataset Description
-----------------------

The experiments were conducted using the NASA CM1 software defect dataset.The dataset consists of software module-level static code metrics and corresponding defect labels.

Each instance represents a software module characterized by various complexity and size-related metrics, including:

*   Lines of Code (LOC)
    
*   Cyclomatic Complexity (v(g))
    
*   Halstead metrics
    
*   Branch Count
    
*   Operator and Operand counts
    
*   Comment density metrics
    

The target variable is binary:

*   **0 (Non-defective)**
    
*   **1 (Defective)**
    

The dataset exhibits class imbalance, with defective modules forming a minority class.

4.2 Data Preprocessing
----------------------

Before model training, the following preprocessing steps were applied:

1.  Feature-target separation.
    
2.  Standardization using StandardScaler.
    
3.  Stratified train-test split (80:20).
    
4.  5-fold Stratified Cross Validation for robustness evaluation.
    

Feature scaling was applied to ensure that models such as Logistic Regression converge effectively and to avoid bias due to magnitude differences among features.

4.3 Model Selection
-------------------

Three machine learning classifiers were evaluated:

1.  Logistic Regression (with class balancing)
    
2.  Random Forest Classifier
    
3.  Gradient Boosting Classifier
    

Hyperparameters were selected based on standard best practices:

*   Logistic Regression:
    
    *   class\_weight = balanced
        
    *   max\_iter = 1000
        
*   Random Forest:
    
    *   n\_estimators = 100
        
    *   class\_weight = balanced
        
*   Gradient Boosting:
    
    *   n\_estimators = 100
        

4.4 Evaluation Metrics
----------------------

The following evaluation metrics were used:

*   Accuracy
    
*   Precision
    
*   Recall (Defect class)
    
*   F1-Score
    
*   Area Under ROC Curve (AUC)
    

Since the dataset is imbalanced, special emphasis was placed on:

*   Recall for the defect class
    
*   ROC-AUC score
    

4.5 Cross-Validation Strategy
-----------------------------

To ensure statistical robustness, 5-fold Stratified Cross Validation was employed.

For each fold:

*   Model training was performed on 80% of data
    
*   Evaluation was conducted on remaining 20%
    
*   AUC and Recall scores were recorded
    

Mean and standard deviation were computed to assess performance stability.

4.6 Statistical Significance Testing
------------------------------------

To compare models rigorously, a paired t-test was conducted between Logistic Regression and Random Forest AUC scores.

The null hypothesis:"There is no statistically significant difference between the model performances."

A p-value < 0.05 was considered statistically significant.

4.7 Feature Importance Analysis
-------------------------------

To interpret model decisions, feature importance analysis was conducted using:

*   Absolute coefficient magnitudes (Logistic Regression)
    

Top influential features were extracted and visualized to understand defect-driving metrics.

This step enhances explainability and supports actionable insights for software quality improvement.