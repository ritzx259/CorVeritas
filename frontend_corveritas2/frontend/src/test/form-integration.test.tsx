import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import Assessment from '../pages/Assessment';
import { emptyDraft } from '../validation/assessment';
import type { Features } from '../types';

const { saveResult, saveDraft, changeModel, resetAssessment } = vi.hoisted(() => ({ saveResult: vi.fn(), saveDraft: vi.fn(), changeModel: vi.fn(), resetAssessment: vi.fn() }));
vi.mock('../state', () => ({
  useResearch: () => ({ models: { default_model_id: 'random_forest', models: [{ id: 'random_forest', name: 'Random Forest', available: true, threshold: 0.5 }] }, loading: false, error: null, refresh: vi.fn() }),
  useAssessment: () => ({ draft: { age: '', sex: '', cp: '', trestbps: '', chol: '', fbs: '', restecg: '', thalch: '', exang: '', oldpeak: '', slope: '', ca: '', thal: '' }, setDraft: saveDraft, setResult: saveResult, selectedModel: 'random_forest', setSelectedModel: changeModel, reset: resetAssessment }),
}));

describe('four-step assessment integration', () => {
  beforeEach(() => { vi.clearAllMocks(); vi.spyOn(window, 'scrollTo').mockImplementation(() => {}); });
  it('validates steps, retains entered fields on Back, and sends exact JSON through the real API client', async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const payload = JSON.parse(init?.body as string) as { model_id: string; features: Features };
      // Transport fixture only. This test verifies the outgoing payload, not model output.
      return new Response(JSON.stringify({ model_id: payload.model_id, predicted_class: 0, predicted_label: 'Fixture response', model_score: 0.25, threshold: 0.5, missing_features: Object.entries(payload.features).filter(([, value]) => value === null).map(([key]) => key), explanation_status: 'unavailable', disclaimer: 'Test fixture' }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<MemoryRouter initialEntries={['/assessment']}><Routes><Route path="/assessment" element={<Assessment />} /><Route path="/results" element={<div>Assessment response received</div>} /></Routes></MemoryRouter>);
    fireEvent.click(screen.getByRole('button', { name: /continue/i }));
    expect(await screen.findByText('Age is required.')).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText(/^age/i), { target: { value: '54' } });
    fireEvent.change(screen.getByLabelText(/^recorded sex/i), { target: { value: 'Female' } });
    fireEvent.click(screen.getByRole('button', { name: /continue/i }));
    await screen.findByRole('heading', { name: 'Symptoms & history' });
    fireEvent.click(screen.getByRole('button', { name: /continue/i }));
    expect(await screen.findByText('Select a chest pain category.')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText(/^chest pain category/i), { target: { value: 'atypical angina' } });
    fireEvent.change(screen.getByLabelText(/^exercise-induced angina/i), { target: { value: 'true' } });
    fireEvent.change(screen.getByLabelText(/^fasting blood sugar above threshold/i), { target: { value: 'false' } });
    fireEvent.click(screen.getByRole('button', { name: /continue/i }));
    await screen.findByRole('heading', { name: 'Clinical measurements' });
    fireEvent.change(screen.getByLabelText(/^thal test result/i), { target: { value: 'reversable defect' } });
    fireEvent.change(screen.getByLabelText(/^major vessels visualized/i), { target: { value: '0' } });
    fireEvent.click(screen.getByRole('button', { name: /^back$/i }));
    await screen.findByRole('heading', { name: 'Symptoms & history' });
    expect(screen.getByLabelText(/^chest pain category/i)).toHaveValue('atypical angina');
    expect(screen.getByLabelText(/^exercise-induced angina/i)).toHaveValue('true');
    fireEvent.click(screen.getByRole('button', { name: /continue/i }));
    await screen.findByRole('heading', { name: 'Clinical measurements' });
    expect(screen.getByLabelText(/^thal test result/i)).toHaveValue('reversable defect');
    fireEvent.click(screen.getByRole('button', { name: /continue/i }));
    await screen.findByRole('heading', { name: 'Review & assess' });
    expect(screen.getByText(/6 of 13 inputs are unavailable/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /run assessment/i }));
    await screen.findByText('Assessment response received');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, request] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/predict');
    const payload = JSON.parse(request?.body as string);
    expect(payload.model_id).toBe('random_forest');
    expect(payload.features).toEqual({ age: 54, sex: 'Female', cp: 'atypical angina', trestbps: null, chol: null, fbs: false, restecg: null, thalch: null, exang: true, oldpeak: null, slope: null, ca: 0, thal: 'reversable defect' });
    await waitFor(() => expect(saveResult).toHaveBeenCalledTimes(1));
    expect(saveResult.mock.calls[0][0].features).toEqual(payload.features);
  });
  it('resets every input to a blank value without requesting a prediction', async () => {
    render(<MemoryRouter><Assessment /></MemoryRouter>);
    fireEvent.change(screen.getByLabelText(/^age/i), { target: { value: '59' } });
    fireEvent.click(screen.getByRole('button', { name: /reset form/i }));
    expect(screen.getByLabelText(/^age/i)).toHaveValue(null);
    expect(screen.getByLabelText(/^recorded sex/i)).toHaveValue('');
    expect(resetAssessment).toHaveBeenCalledOnce();
    await waitFor(() => expect(saveDraft).toHaveBeenLastCalledWith(emptyDraft));
  });
});
