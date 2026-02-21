# 🔍 Cross-Dataset Software Defect Prediction using Machine Learning

A research-grade and production-ready AI system for predicting defect-prone software modules using static code metrics.

This repository accompanies the research publication:

**"Cross-Dataset Evaluation of Imbalance Handling Strategies for Software Defect Prediction Using Machine Learning"**

📄 DOI: https://zenodo.org/records/18726045
Deploy link: https://defect-risk-ai-service.onrender.com
👨‍💻 Author: Dhruv Sharma  

---

## 🚀 Project Overview

Software defect prediction plays a critical role in risk-based testing and intelligent test automation. However, real-world defect datasets are highly imbalanced, making it difficult to detect defective modules accurately.

This project:

- Evaluates multiple NASA MDP datasets (CM1, KC1, PC1, JM1)
- Compares imbalance handling strategies:
  - No balancing
  - Class-weight balancing
  - SMOTE
- Performs cross-dataset evaluation
- Conducts statistical significance testing
- Deploys the trained model as a production-ready FastAPI service
- Dockerizes the complete AI system

---

## 📊 Key Findings

Across four datasets:

- ❌ No balancing → Very low recall (0.06–0.20)
- ⚖ Class-weight balancing → Highest recall (~0.60–0.72)
- 🔁 SMOTE → Competitive but slightly lower recall

Class-weight balancing consistently achieved the best trade-off between AUC and defect recall.

This highlights the importance of imbalance-aware learning in real-world defect prediction systems.

---

## 🧠 Datasets Used

NASA Metrics Data Program (MDP) datasets:

- CM1
- KC1
- PC1
- JM1

These datasets contain static code metrics including:
- Lines of Code (LOC)
- Cyclomatic Complexity
- Halstead metrics
- Branch count
- Code/comment metrics

---


---

## ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/YOUR_USERNAME/defect-risk-ai-service.git
cd defect-risk-ai-service

Install dependencies:

pip install -r requirements.txt
🧪 Run Experiments

Single-dataset experiment:

python ml/experiments.py

Multi-dataset experiment:

python ml/multi_dataset_experiments.py

Imbalance strategy comparison:

python ml/multi_dataset_smote_experiment.py
🧮 Train Final Model
python ml/train.py

This will generate:

artifacts/model.pkl
artifacts/scaler.pkl
🌐 Run API Locally
uvicorn app.main:app --reload

Open Swagger UI:

http://127.0.0.1:8000/docs
🐳 Docker Deployment

Build Docker image:

docker build -t defect-risk-ai .

Run container:

docker run -p 8000:8000 defect-risk-ai
☁️ Cloud Deployment

The system has been successfully deployed on Render using Docker containerization.

📈 Evaluation Metrics

Area Under ROC Curve (AUC)

Recall (Defect class)

5-Fold Cross Validation

Paired t-test for statistical significance

🎓 Research Contribution

This study provides:

Cross-dataset validation

Empirical imbalance strategy comparison

Statistical performance validation

Production-ready ML deployment

Integration pathway toward intelligent test automation

📌 Future Work

Cross-project transfer learning

Cost-sensitive learning

Explainable AI (SHAP)

Deep learning-based defect prediction

📜 License

This project is licensed under the MIT License.

📬 Contact

Dhruv Sharma
