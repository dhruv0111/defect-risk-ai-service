6\. Conclusion and Research Contributions
=========================================

6.1 Conclusion
--------------

This study investigated the application of machine learning techniques for software defect prediction using static code metrics from the NASA CM1 dataset. Three classification models—Logistic Regression, Random Forest, and Gradient Boosting—were evaluated under both single train-test split and 5-fold stratified cross-validation settings.

The experimental results demonstrate that Logistic Regression outperformed ensemble-based models in terms of defect recall and cross-validated AUC performance. While ensemble methods achieved competitive accuracy, they failed to consistently detect defect-prone modules under severe class imbalance. Logistic Regression achieved a mean AUC of 0.8023 and a mean defect recall of 0.7289 across 5 folds, significantly outperforming Random Forest based on paired t-test analysis (p < 0.05).

These findings suggest that simpler linear models may generalize more effectively than complex ensemble methods in small, imbalanced software defect datasets. The results also highlight the importance of prioritizing recall-oriented evaluation metrics when designing defect prediction systems for practical software testing environments.

Feature importance analysis revealed that complexity-related and size-related metrics—such as Lines of Code, Cyclomatic Complexity, Unique Operators, and Branch Count—play a significant role in defect prediction. These findings align with established software engineering principles regarding code complexity and defect proneness.

Overall, the study demonstrates that interpretable machine learning models can provide both predictive capability and actionable insights for improving software quality.

6.2 Research Contributions
--------------------------

The primary contributions of this study are summarized as follows:

1.  Comparative Experimental AnalysisA systematic comparison of linear and ensemble machine learning models for software defect prediction under imbalanced conditions.
    
2.  Robust Cross-Validation EvaluationImplementation of 5-fold stratified cross-validation to ensure statistical stability and reduce sampling bias.
    
3.  Statistical Significance VerificationUse of paired t-test analysis to confirm performance differences between competing models.
    
4.  Recall-Oriented Evaluation StrategyEmphasis on defect-class recall as a primary evaluation metric to align model performance with real-world testing priorities.
    
5.  Feature Interpretability AssessmentExtraction and analysis of influential software metrics to provide insights into defect-driving characteristics.
    
6.  Reproducible Experimental PipelineDevelopment of a structured, production-ready experimental framework capable of integration into real-world software testing systems.
    

6.3 Future Work
---------------

Future research directions may include:

*   Evaluating deep learning models such as neural networks for defect prediction.
    
*   Applying graph-based learning techniques to model code structure.
    
*   Exploring transfer learning across multiple NASA datasets.
    
*   Incorporating dynamic runtime metrics alongside static code metrics.
    
*   Integrating defect prediction models into continuous integration pipelines for real-time risk assessment.
    

Extending this work toward larger multi-project datasets could further validate model generalization capabilities and improve predictive robustness.