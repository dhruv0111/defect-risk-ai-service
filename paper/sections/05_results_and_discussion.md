5\. Results and Discussion
==========================

5.1 Train-Test Evaluation Results
---------------------------------

Initial experiments were conducted using an 80–20 stratified train-test split. The performance comparison of the evaluated models is summarized in Table 1.

Logistic Regression achieved an AUC score of 0.6589, while Random Forest and Gradient Boosting achieved AUC scores of 0.6594 and 0.6550 respectively. Although the AUC values were similar across models, significant differences were observed in recall for the defect class.

Logistic Regression achieved a recall of 0.50 for defective modules, compared to 0.10 for Random Forest and 0.20 for Gradient Boosting. This indicates that Logistic Regression was more effective in identifying defect-prone modules under imbalanced conditions.

Given that missing defective modules is more costly in real-world testing scenarios, recall for the defect class is considered a critical evaluation metric.

5.2 Cross-Validation Performance
--------------------------------

To ensure robustness, 5-fold Stratified Cross Validation was performed.

The results demonstrated:

*   Logistic Regression:
    
    *   Mean AUC = 0.8023 ± 0.0963
        
    *   Mean Recall (Defect) = 0.7289
        
*   Random Forest:
    
    *   Mean AUC = 0.7353 ± 0.0907
        
    *   Mean Recall (Defect) = 0.0
        
*   Gradient Boosting:
    
    *   Mean AUC = 0.6673 ± 0.1322
        
    *   Mean Recall (Defect) = 0.08
        

The cross-validation results reveal a substantial improvement in Logistic Regression performance compared to the single split evaluation. The mean AUC of 0.80 suggests strong discriminative capability when evaluated across multiple folds.

Notably, Random Forest failed to detect defective modules consistently across folds, yielding a mean recall of 0.0. This behavior may be attributed to the limited dataset size and severe class imbalance, where ensemble models may bias predictions toward the majority class.

5.3 Statistical Significance Analysis
-------------------------------------

A paired t-test was conducted to compare the AUC scores of Logistic Regression and Random Forest across the 5 folds.

The results showed:

*   T-statistic = 3.2569
    
*   P-value = 0.0312
    

Since the p-value is less than 0.05, the null hypothesis is rejected, indicating that Logistic Regression significantly outperforms Random Forest in terms of AUC.

This statistical validation strengthens the reliability of the experimental findings.

5.4 Feature Importance Analysis
-------------------------------

Feature importance analysis based on Logistic Regression coefficients revealed that the most influential predictors include:

*   Comment-related metrics (lOComment)
    
*   Unique operator count (uniq\_Op)
    
*   Lines of code (loc)
    
*   Cyclomatic complexity (v(g))
    
*   Branch count (branchCount)
    

These findings align with established software engineering principles, which suggest that larger and more complex modules are more likely to contain defects.

The presence of comment-related metrics among top predictors suggests that documentation patterns may correlate with defect risk, potentially reflecting development practices or code maintainability characteristics.

5.5 Practical Implications
--------------------------

The results suggest that:

1.  Simpler linear models may generalize better than complex ensemble methods on small, imbalanced software defect datasets.
    
2.  Recall-oriented optimization is crucial for practical defect prediction systems.
    
3.  Feature interpretability provides actionable insights for improving software quality practices.
    

From a software testing perspective, integrating Logistic Regression-based defect prediction into test automation pipelines can enable risk-based prioritization of modules, thereby improving testing efficiency and reducing defect leakage.