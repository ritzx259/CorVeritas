from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt


Sex = Literal["Male", "Female"]
ChestPain = Literal[
    "typical angina",
    "atypical angina",
    "non-anginal",
    "asymptomatic",
]
RestingEcg = Literal["normal", "st-t abnormality", "lv hypertrophy"]
Slope = Literal["upsloping", "flat", "downsloping"]
Thal = Literal["normal", "fixed defect", "reversable defect"]


class Features(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: StrictInt = Field(ge=0, le=120)
    sex: Sex
    cp: ChestPain
    trestbps: Optional[StrictFloat] = Field(default=None, ge=0, le=400)
    chol: Optional[StrictFloat] = Field(default=None, ge=0, le=1500)
    fbs: Optional[StrictBool] = None
    restecg: Optional[RestingEcg] = None
    thalch: Optional[StrictFloat] = Field(default=None, ge=0, le=350)
    exang: Optional[StrictBool] = None
    oldpeak: Optional[StrictFloat] = Field(default=None, ge=-20, le=20)
    slope: Optional[Slope] = None
    ca: Optional[StrictInt] = Field(default=None, ge=0, le=3)
    thal: Optional[Thal] = None

    def ordered_dict(self):
        return self.model_dump()


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    features: Features


class Contribution(BaseModel):
    feature: str
    contribution: float


class PredictionResponse(BaseModel):
    model_id: str
    predicted_class: int
    predicted_label: str
    model_score: float
    threshold: float
    missing_features: list[str]
    explanation_status: str
    explanations: Optional[list[Contribution]] = None
    disclaimer: str
