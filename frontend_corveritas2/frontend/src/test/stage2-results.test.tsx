import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import Results, { ResultContent } from '../pages/Results';
import Stage2Boundary from '../components/stage2/Stage2Boundary';
import { stage2Api } from '../api/stage2';
import { emptyDraft, toFeatures } from '../validation/assessment';
import type { AssessmentResult } from '../types';

const { reset } = vi.hoisted(() => ({ reset: vi.fn() }));
const result: AssessmentResult = { modelName: 'Original Stage 1 model', features: toFeatures({ ...emptyDraft, age: '55', sex: 'Male', cp: 'asymptomatic' }), response: { model_id: 'random_forest', predicted_class: 1, predicted_label: 'Disease-associated pattern detected', model_score: 0.8, threshold: 0.5, missing_features: [], explanation_status: 'unavailable', disclaimer: 'Original disclaimer' } };
vi.mock('../state', () => ({ useAssessment: () => ({ result, reset }) }));
vi.mock('../api/stage2', () => ({ stage2Api: { schema: vi.fn(), predict: vi.fn() } }));

afterEach(() => { vi.restoreAllMocks(); });
describe('Stage 1 rendering remains independent of Stage 2', () => {
  it('renders the original result while Stage 2 metadata is still loading', async () => {
    vi.spyOn(window, 'scrollTo').mockImplementation(() => {});
    vi.mocked(stage2Api.schema).mockReturnValue(new Promise(() => {}));
    render(<MemoryRouter><Results /></MemoryRouter>);
    expect(screen.getByText('80.0')).toBeInTheDocument();
    expect(screen.getByText('Original Stage 1 model')).toBeInTheDocument();
    expect(await screen.findByText('Checking the isolated Stage 2 model…')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Stage 1 — Cardiovascular Disease Detection' })).toBeInTheDocument();
    expect(stage2Api.predict).not.toHaveBeenCalled();
  });
  it('contains a Stage 2 render failure without removing the existing score or explanation section', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    function BrokenStage2(): never { throw new Error('Stage 2 fixture render failure'); }
    render(<MemoryRouter><ResultContent result={result} /><Stage2Boundary><BrokenStage2 /></Stage2Boundary></MemoryRouter>);
    expect(screen.getByText('80.0')).toBeInTheDocument();
    expect(screen.getByText('Original Stage 1 model')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'What contributed to this result?' })).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('Stage 2 is unavailable. Your Stage 1 result is preserved.');
  });
});
