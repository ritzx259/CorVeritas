export type PatientFeatures = {
  age: number;
  sex: "Male" | "Female";
  cp: "typical angina" | "atypical angina" | "non-anginal" | "asymptomatic";
  trestbps: number | null;
  chol: number | null;
  fbs: boolean | null;
  restecg: "normal" | "st-t abnormality" | "lv hypertrophy" | null;
  thalch: number | null;
  exang: boolean | null;
  oldpeak: number | null;
  slope: "upsloping" | "flat" | "downsloping" | null;
  ca: number | null;
  thal: "normal" | "fixed defect" | "reversable defect" | null;
};

export type ModelInfo = {
  id: string;
  name: string;
  architecture: string;
  description: string;
  available: boolean;
  error: string | null;
  is_default: boolean;
  threshold: number;
  metrics: Record<string, number | null>;
  metrics_source: string;
  graphs: string[];
  input_features: string[];
  explanation: string;
};

export type ModelsResponse = {
  default_model_id: string | null;
  models: ModelInfo[];
  disclaimer: string;
};

export type Prediction = {
  model_id: string;
  predicted_class: 0 | 1;
  predicted_label: string;
  model_score: number;
  threshold: number;
  missing_features: string[];
  explanation_status: string;
  explanations?: { feature: string; contribution: number }[] | null;
  disclaimer: string;
};

export type Health = {
  status: "ok" | "degraded";
  ready: boolean;
  available_model_ids: string[];
  model_errors: Record<string, string>;
};
