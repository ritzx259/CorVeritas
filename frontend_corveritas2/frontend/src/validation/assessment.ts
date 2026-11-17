import { z } from 'zod';
import type { AssessmentDraft, Features, FeatureName } from '../types';

const numberField = (label: string, min: number, max: number, integer = false, required = false) => z.string().superRefine((value, context) => {
  if (value.trim() === '') { if (required) context.addIssue({ code: 'custom', message: `${label} is required.` }); return; }
  const number = Number(value);
  if (!Number.isFinite(number)) context.addIssue({ code: 'custom', message: 'Enter a valid number.' });
  else if (integer && !Number.isInteger(number)) context.addIssue({ code: 'custom', message: 'Enter a whole number.' });
  else if (number < min || number > max) context.addIssue({ code: 'custom', message: `Enter a value between ${min} and ${max}.` });
});
export const assessmentSchema = z.object({
  age: numberField('Age', 18, 120, true, true),
  sex: z.enum(['Male', 'Female'], { message: 'Select the sex recorded for the assessment.' }),
  cp: z.enum(['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'], { message: 'Select a chest pain category.' }),
  trestbps: numberField('Resting blood pressure', 40, 300),
  chol: numberField('Cholesterol', 0, 1000),
  fbs: z.enum(['', 'true', 'false']),
  restecg: z.enum(['', 'normal', 'st-t abnormality', 'lv hypertrophy']),
  thalch: numberField('Maximum heart rate', 30, 250),
  exang: z.enum(['', 'true', 'false']),
  oldpeak: numberField('ST depression', -5, 15),
  slope: z.enum(['', 'upsloping', 'flat', 'downsloping']),
  ca: numberField('Major vessels', 0, 3, true),
  thal: z.enum(['', 'normal', 'fixed defect', 'reversable defect']),
});
export const emptyDraft: AssessmentDraft = { age: '', sex: '', cp: '', trestbps: '', chol: '', fbs: '', restecg: '', thalch: '', exang: '', oldpeak: '', slope: '', ca: '', thal: '' };
export const featureOrder: FeatureName[] = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'];
export function toFeatures(draft: AssessmentDraft): Features {
  assessmentSchema.parse(draft);
  return Object.fromEntries(featureOrder.map((key) => {
    const value = draft[key];
    if (value.trim() === '') return [key, null];
    if (['fbs', 'exang'].includes(key)) return [key, value === 'true'];
    if (['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca'].includes(key)) return [key, Number(value)];
    return [key, value];
  })) as Features;
}
export const fieldLabels: Record<FeatureName, string> = {
  age: 'Age', sex: 'Recorded sex', cp: 'Chest pain category', trestbps: 'Resting blood pressure', chol: 'Serum cholesterol', fbs: 'Fasting blood sugar > 120 mg/dL', restecg: 'Resting ECG', thalch: 'Maximum heart rate', exang: 'Exercise-induced angina', oldpeak: 'ST depression', slope: 'ST segment slope', ca: 'Major vessels visualized', thal: 'Thal test result',
};
export const fieldUnits: Partial<Record<FeatureName, string>> = { age: 'years', trestbps: 'mmHg', chol: 'mg/dL', thalch: 'bpm', oldpeak: 'mm' };
export function formatValue(name: FeatureName, value: Features[FeatureName] | undefined): string {
  if (value === null || value === undefined || value === '') return 'Unknown / not available';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (value === 'reversable defect') return 'Reversible defect';
  if (value === 'lv hypertrophy') return 'Left ventricular hypertrophy';
  const text = String(value);
  return `${text.charAt(0).toUpperCase()}${text.slice(1)}${fieldUnits[name] ? ` ${fieldUnits[name]}` : ''}`;
}
