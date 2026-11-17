import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import { ResultContent } from '../pages/Results';
import { emptyDraft, toFeatures } from '../validation/assessment';
import type { AssessmentResult } from '../types';

// A response fixture verifies display behavior; it is never used for inference.
const result: AssessmentResult = {
  modelName: 'Test neural model',
  features: toFeatures({ ...emptyDraft, age: '54', sex: 'Female', cp: 'atypical angina' }),
  response: { model_id: 'test', predicted_class: 0, predicted_label: 'No disease-associated pattern detected', model_score: 0.23, threshold: 0.5, missing_features: ['chol'], explanation_status: 'unavailable', disclaimer: 'Educational research use only.' },
};
describe('honest result presentation', () => {
  it('renders the actual response score, threshold, missing inputs and medical limitations', () => {
    render(<MemoryRouter><ResultContent result={result} /></MemoryRouter>);
    expect(screen.getByText('No disease-associated pattern detected')).toBeInTheDocument();
    expect(screen.getByText('23.0')).toBeInTheDocument();
    expect(screen.getByText('Threshold 50%')).toBeInTheDocument();
    expect(screen.getByText(/does not exclude heart disease/)).toBeInTheDocument();
    expect(screen.getByText(/not a verified clinical risk probability/)).toBeInTheDocument();
    expect(screen.getByText('Explanation currently unavailable.')).toBeInTheDocument();
    expect(screen.getByText(/1 missing inputs handled/)).toBeInTheDocument();
    expect(screen.getByText('Test neural model')).toBeInTheDocument();
  });
  it('does not turn a positive class into a disease-subtype diagnosis', () => {
    render(<MemoryRouter><ResultContent result={{ ...result, response: { ...result.response, predicted_class: 1, predicted_label: 'Disease-associated pattern detected', model_score: 0.8 } }} /></MemoryRouter>);
    expect(screen.getByText('Disease-associated pattern detected')).toBeInTheDocument();
    expect(screen.getByText(/does not establish a diagnosis/)).toBeInTheDocument();
    expect(screen.queryByText(/myocardial infarction/i)).not.toBeInTheDocument();
  });
});
