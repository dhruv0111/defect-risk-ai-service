Abstract
========

Software defect prediction plays a critical role in improving software quality and reducing testing costs. However, real-world defect datasets often exhibit severe class imbalance, making accurate detection of defect-prone modules challenging. This study presents a recall-oriented comparative evaluation of machine learning models for imbalanced software defect prediction using the NASA CM1 dataset.

Three classification models—Logistic Regression, Random Forest, and Gradient Boosting—were evaluated using both stratified train-test splits and 5-fold cross-validation. Experimental results demonstrate that Logistic Regression significantly outperforms ensemble-based methods in terms of defect recall and cross-validated AUC performance. The model achieved a mean AUC of 0.80 and a mean defect recall of 0.73 across folds, with statistical significance confirmed via paired t-test analysis (p < 0.05).

Feature importance analysis revealed that size- and complexity-related metrics, including Lines of Code and Cyclomatic Complexity, are strong predictors of defect proneness. The findings suggest that simpler linear models may generalize more effectively than complex ensemble approaches in small, imbalanced software defect datasets.

The study emphasizes recall-oriented evaluation strategies and provides a reproducible experimental framework suitable for integration into risk-based testing systems.