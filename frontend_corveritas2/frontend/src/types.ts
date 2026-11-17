export type FeatureName = 'age' | 'sex' | 'cp' | 'trestbps' | 'chol' | 'fbs' | 'restecg' | 'thalch' | 'exang' | 'oldpeak' | 'slope' | 'ca' | 'thal';
export type Features = Record<FeatureName, number | string | boolean | null>;
export type AssessmentDraft = Record<FeatureName, string>;
export interface ModelInfo {
  id: string; name: string; algorithm?: string; available: boolean; reason?: string | null;
  threshold: number; architecture?: string | string[]; description?: string;
  metrics?: Record<string, number | null> | null; metrics_source?: string | null; metrics_note?: string;
  metrics_evaluation?: { partition: 'validation' | 'test'; sample_count: number; threshold: number; used_during_training: boolean } | null;
  artifact_directory?: string;
  plots?: { name: string; url: string }[]; training_command?: string;
  explanation_method?: string | null;
  global_importance?: { feature: string; importance: number }[];
}
export interface ModelsResponse { models: ModelInfo[]; default_model_id: string | null }
export interface Health { status: string; ready: boolean; loaded_models?: number | string[]; message?: string }
export interface SchemaField {
  name: FeatureName; label: string; type: string; required: boolean;
  unit?: string | null; categories?: { value: string | boolean; label: string }[] | null; min?: number | null; max?: number | null;
}
export interface InputSchema { fields: SchemaField[]; feature_order?: string[]; dataset?: Record<string, unknown> }
export interface Contributions { feature: string; label: string; value: number }
export interface Explanation {
  method: string; description?: string; baseline?: number; score?: number; reconstruction_error?: number;
  contributions?: Contributions[]; global_importance?: Contributions[];
  note?: string; units?: string;
}
export interface Prediction {
  model_id: string; predicted_class: 0 | 1; predicted_label: string; model_score: number;
  threshold: number; missing_features: string[]; explanation_status: string;
  explanations?: Explanation | null; disclaimer: string;
}
export interface AssessmentResult { response: Prediction; features: Features; modelName: string }
