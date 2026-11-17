import type { AssessmentResult, FeatureName } from '../../types';

export interface Stage2Field {
  name: string; label: string; unit: string | null; help: string; required: boolean;
  type: 'number' | 'category'; min?: number; max?: number; categories?: string[];
  source_stage1?: FeatureName | null;
}
export interface Stage2Schema {
  available: boolean; reason?: string; model?: string; model_name?: string; run_id?: string;
  model_version?: string; features?: Stage2Field[]; conditions?: { id: string; name: string }[];
  output_type?: string; explanation_supported?: boolean; limitations?: string[];
}
export interface Stage2Condition {
  condition: string; condition_name: string; model_score: number; selected_threshold: number;
  binary_classification: 0 | 1; score_type: string; classification_meaning: string;
  feature_contributions: { feature: string; contribution: number; description: string }[];
  explanation_base_score?: number; explanation_reconstruction_error?: number;
}
export interface Stage2Prediction {
  model: string; model_name: string; model_version: string; run_id: string;
  conditions: Stage2Condition[]; warnings: string[]; limitations: string[];
  explanation_method: string | null; explanation_limitations: string | null;
}
export type Stage2Values = Record<string, string | number | null>;

export function reusedStage1Values(result: AssessmentResult, fields: Stage2Field[]): Stage2Values {
  const values: Stage2Values = {};
  for (const field of fields) {
    const value = field.source_stage1 ? result.features[field.source_stage1] : null;
    if (value == null || typeof value === 'boolean') continue;
    if (field.type === 'number' && (typeof value !== 'number' || value < (field.min ?? -Infinity) || value > (field.max ?? Infinity))) continue;
    if (field.type === 'category' && !field.categories?.includes(String(value))) continue;
    values[field.name] = value;
  }
  return values;
}

export function stage2Inputs(fields: Stage2Field[], reused: Stage2Values, draft: Record<string, string>) {
  const additional: Stage2Values = {};
  const errors: Record<string, string> = {};
  for (const field of fields) {
    if (field.name in reused) continue;
    const raw = (draft[field.name] ?? '').trim();
    if (!raw) {
      additional[field.name] = null;
      if (field.required) errors[field.name] = `${field.label} is required.`;
    } else if (field.type === 'number') {
      const number = Number(raw);
      if (!Number.isFinite(number) || number < (field.min ?? -Infinity) || number > (field.max ?? Infinity)) errors[field.name] = `${field.label} must be between ${field.min} and ${field.max}${field.unit ? ` ${field.unit}` : ''}.`;
      else additional[field.name] = number;
    } else if (!field.categories?.includes(raw)) errors[field.name] = `Select a supported ${field.label.toLowerCase()} value.`;
    else additional[field.name] = raw;
  }
  const combined = { ...reused, ...additional };
  if (typeof combined.systolic_bp === 'number' && typeof combined.diastolic_bp === 'number' && combined.systolic_bp <= combined.diastolic_bp) errors.diastolic_bp = 'Diastolic pressure must be below systolic pressure.';
  return { additional, errors };
}
