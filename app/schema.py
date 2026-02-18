# app/schema.py

# app/schema.py

from pydantic import BaseModel, Field
from typing import List


class PredictionInput(BaseModel):
    features: List[float] = Field(
        ...,
        example=[
            1.4,1.4,1.4,1.3,1.3,1.3,1.3,1.3,1.3,1.3,
            2,2,2,2,1.2,1.2,1.2,1.2,1.4,2,3
        ]
    )


class PredictionOutput(BaseModel):
    defect_probability: float
    risk_level: str
