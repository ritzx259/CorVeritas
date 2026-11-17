import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import Stage2Panel from '../components/stage2/Stage2Panel';
import { reusedStage1Values, stage2Inputs, type Stage2Prediction, type Stage2Schema } from '../components/stage2/types';
import { stage2Api } from '../api/stage2';
import type { AssessmentResult } from '../types';
import { emptyDraft, toFeatures } from '../validation/assessment';

vi.mock('../api/stage2', () => ({ stage2Api: { schema: vi.fn(), predict: vi.fn() } }));
const result: AssessmentResult = { modelName: 'Stage 1 fixture', features: toFeatures({ ...emptyDraft, age: '55', sex: 'Male', cp: 'asymptomatic', trestbps: '145', chol: '240', fbs: 'true' }), response: { model_id: 'random_forest', predicted_class: 1, predicted_label: 'Disease-associated pattern detected', model_score: 0.8, threshold: 0.5, missing_features: [], explanation_status: 'unavailable', disclaimer: 'Fixture' } };
const schema: Stage2Schema = {
  available: true, model: 'neural', model_name: 'Stage 2 multi-label MLP', run_id: 'fixture-run',
  conditions: [{ id: 'coronary_heart_disease', name: 'Reported coronary heart disease history' }, { id: 'heart_failure', name: 'Reported congestive heart failure history' }, { id: 'heart_attack_history', name: 'Reported previous myocardial infarction' }],
  features: [
    { name: 'age', label: 'Age', type: 'number', min: 20, max: 120, required: true, unit: 'years', help: '20+', source_stage1: 'age' },
    { name: 'sex', label: 'Recorded sex', type: 'category', categories: ['Male', 'Female', 'Unknown'], required: false, unit: null, help: '', source_stage1: 'sex' },
    { name: 'systolic_bp', label: 'Systolic blood pressure', type: 'number', min: 50, max: 300, required: false, unit: 'mmHg', help: '', source_stage1: 'trestbps' },
    { name: 'total_cholesterol', label: 'Total cholesterol', type: 'number', min: 40, max: 1000, required: false, unit: 'mg/dL', help: '', source_stage1: 'chol' },
    { name: 'bmi', label: 'Body mass index', type: 'number', min: 10, max: 100, required: false, unit: 'kg/m²', help: 'Documented BMI' },
    { name: 'diastolic_bp', label: 'Diastolic blood pressure', type: 'number', min: 20, max: 200, required: false, unit: 'mmHg', help: '' },
    { name: 'diabetes_history', label: 'Reported diabetes history', type: 'category', categories: ['Yes', 'No', 'Borderline', 'Unknown'], required: false, unit: null, help: 'Not fasting glucose' },
    { name: 'smoking_history', label: '100+ lifetime cigarettes', type: 'category', categories: ['Yes', 'No', 'Unknown'], required: false, unit: null, help: 'Lifetime threshold' },
  ],
};
const prediction: Stage2Prediction = { model: 'neural', model_name: 'Stage 2 multi-label MLP', model_version: 'fixture', run_id: 'fixture-run', warnings: [], limitations: ['Fixture limitation'], explanation_method: null, explanation_limitations: null, conditions: schema.conditions!.map((condition, index) => ({ condition: condition.id, condition_name: condition.name, model_score: index === 0 ? 0.25 : 0.1, selected_threshold: 0.15, binary_classification: index === 0 ? 1 : 0, score_type: 'uncalibrated_model_score', classification_meaning: 'Research threshold', feature_contributions: [] })) };
const mount = (value = result) => render(<MemoryRouter><p>Preserved Stage 1 result: 80.0%</p><Stage2Panel result={value} /></MemoryRouter>);

describe('Stage 2 extension', () => {
  beforeEach(() => { vi.resetAllMocks(); vi.mocked(stage2Api.schema).mockResolvedValue(schema); vi.mocked(stage2Api.predict).mockResolvedValue(prediction); });
  it('does not load or run Stage 2 after a negative Stage 1 result', () => {
    mount({ ...result, response: { ...result.response, predicted_class: 0 } });
    expect(screen.getByText('Disease type classification was not performed because Stage 1 did not indicate cardiovascular disease.')).toBeInTheDocument();
    expect(screen.getByText(/does not guarantee that disease is absent/)).toBeInTheDocument();
    expect(stage2Api.schema).not.toHaveBeenCalled(); expect(stage2Api.predict).not.toHaveBeenCalled();
  });
  it('keeps Stage 1 eligibility for 18–19 separate from the Stage 2 adult range', () => {
    mount({ ...result, features: { ...result.features, age: 19 } });
    expect(screen.getByText('Stage 2 is available only for adults aged 20–120.')).toBeInTheDocument();
    expect(stage2Api.schema).not.toHaveBeenCalled();
  });
  it('requests only missing inputs, classifies only on click, and shows all independent scores', async () => {
    mount();
    await screen.findByText('Ready to classify');
    expect(screen.queryByLabelText(/^Age/)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/^Systolic blood pressure/)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/^Total cholesterol/)).not.toBeInTheDocument();
    expect(screen.getByLabelText(/^Reported diabetes history/)).toHaveValue('');
    expect(stage2Api.predict).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText(/^Body mass index/), { target: { value: '28.4' } });
    fireEvent.change(screen.getByLabelText(/^Reported diabetes history/), { target: { value: 'Borderline' } });
    fireEvent.click(screen.getByRole('button', { name: 'Classify Disease Type' }));
    await screen.findByText('Classification completed');
    expect(stage2Api.predict).toHaveBeenCalledWith(result, { bmi: 28.4, diastolic_bp: null, diabetes_history: 'Borderline', smoking_history: null });
    expect(screen.getByText('25.0%')).toBeInTheDocument();
    expect(screen.getAllByText('10.0%')).toHaveLength(2);
    expect(screen.getAllByRole('img', { name: /model score .* threshold/ })).toHaveLength(3);
    expect(screen.getByText(/scores do not sum to 100%/)).toBeInTheDocument();
    expect(screen.getByText('Preserved Stage 1 result: 80.0%')).toBeInTheDocument();
  });
  it('blocks inconsistent blood pressure without calling the classifier', async () => {
    mount(); await screen.findByText('Ready to classify');
    fireEvent.change(screen.getByLabelText(/^Diastolic blood pressure/), { target: { value: '150' } });
    fireEvent.click(screen.getByRole('button', { name: 'Classify Disease Type' }));
    expect(await screen.findByText('Diastolic pressure must be below systolic pressure.')).toBeInTheDocument();
    expect(stage2Api.predict).not.toHaveBeenCalled();
  });
  it('allows unknown optional values and uses the real Stage 2 imputation path', async () => {
    mount(); await screen.findByText('Ready to classify');
    fireEvent.click(screen.getByRole('button', { name: 'Classify Disease Type' }));
    await screen.findByText('Classification completed');
    expect(stage2Api.predict).toHaveBeenCalledWith(result, { bmi: null, diastolic_bp: null, diabetes_history: null, smoking_history: null });
  });
  it('does not force a winning disease category when every score is below threshold', async () => {
    vi.mocked(stage2Api.predict).mockResolvedValue({ ...prediction, conditions: prediction.conditions.map(condition => ({ ...condition, model_score: 0.1, binary_classification: 0 })) });
    mount(); await screen.findByText('Ready to classify');
    fireEvent.click(screen.getByRole('button', { name: 'Classify Disease Type' }));
    expect(await screen.findByText(/No supported category exceeded its saved research threshold/)).toBeInTheDocument();
    expect(screen.getAllByText('Below threshold')).toHaveLength(3);
  });
  it('keeps Stage 1 visible while running and after a Stage 2 failure', async () => {
    let reject!: (reason: Error) => void;
    vi.mocked(stage2Api.predict).mockReturnValue(new Promise((_, failure) => { reject = failure; }));
    mount(); await screen.findByText('Ready to classify');
    fireEvent.click(screen.getByRole('button', { name: 'Classify Disease Type' }));
    expect(screen.getByText('Classification in progress')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Classifying/ })).toBeDisabled();
    expect(screen.getByText('Preserved Stage 1 result: 80.0%')).toBeInTheDocument();
    reject(new Error('Stage 2 timed out'));
    await screen.findByText('Classification failed');
    expect(screen.getByRole('alert')).toHaveTextContent('Stage 2 timed out');
    expect(screen.getByText('Preserved Stage 1 result: 80.0%')).toBeInTheDocument();
  });
  it('can retry Stage 2 availability without resubmitting Stage 1', async () => {
    vi.mocked(stage2Api.schema).mockResolvedValueOnce({ available: false, reason: 'Stage 2 files missing' }).mockResolvedValueOnce(schema);
    mount(); await screen.findByText('Classification unavailable');
    fireEvent.click(screen.getByRole('button', { name: 'Retry Stage 2 availability' }));
    await screen.findByText('Ready to classify');
    expect(stage2Api.predict).not.toHaveBeenCalled();
    expect(screen.getByText('Preserved Stage 1 result: 80.0%')).toBeInTheDocument();
  });
  it('does not reuse out-of-range Stage 1 values or infer diagnosis history from fasting glucose', () => {
    const reused = reusedStage1Values({ ...result, features: { ...result.features, trestbps: 40, chol: 0 } }, schema.features!);
    expect(reused).toEqual({ age: 55, sex: 'Male' });
    const inputs = stage2Inputs(schema.features!, reused, {});
    expect(inputs.additional).toMatchObject({ systolic_bp: null, total_cholesterol: null, diabetes_history: null });
  });
  it('supports a required field declared by the supplied schema', async () => {
    vi.mocked(stage2Api.schema).mockResolvedValue({ ...schema, features: schema.features!.map(field => field.name === 'bmi' ? { ...field, required: true } : field) });
    mount(); await screen.findByText('Missing required inputs');
    expect(screen.getByRole('button', { name: 'Classify Disease Type' })).toBeDisabled();
    fireEvent.change(screen.getByLabelText(/^Body mass index/), { target: { value: '26' } });
    await waitFor(() => expect(screen.getByRole('button', { name: 'Classify Disease Type' })).toBeEnabled());
  });
});
