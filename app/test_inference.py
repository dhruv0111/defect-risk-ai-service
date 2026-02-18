# app/test_inference.py

from app.model import DefectRiskModel


# Example fake input (must match number of features)
# CM1 has 21 features after removing id & defects
sample_input = [1.4, 1.4, 1.4, 1.3, 1.3, 1.3, 1.3, 1.3,
                1.3, 1.3, 2, 2, 2, 2, 1.2, 1.2, 1.2, 1.2,
                1.4, 2, 3]  # adjust length if needed


model = DefectRiskModel()
result = model.predict(sample_input)

print(result)
