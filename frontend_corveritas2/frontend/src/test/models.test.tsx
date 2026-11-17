import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import Models from '../pages/Models';

vi.mock('../state', () => ({ useResearch: () => ({
  loading: false, error: null, refresh: vi.fn(), models: { default_model_id: 'mlp', models: [
    { id: 'mlp', name: 'Test MLP', available: true, threshold: 0.5,
      metrics: { accuracy: 0.812345, precision: 0, recall: 0.7, f1: 0.75, roc_auc: 0.88, average_precision: 0.85 },
      metrics_evaluation: { partition: 'validation', sample_count: 184, threshold: 0.5, used_during_training: true },
      metrics_note: 'Saved validation evidence for the selected artifact.' },
    { id: 'cnn', name: 'Test CNN', available: true, threshold: 0.5,
      metrics: { accuracy: 0.84 },
      metrics_evaluation: { partition: 'test', sample_count: 184, threshold: 0.5, used_during_training: false },
      metrics_note: 'Saved test evidence for the selected artifact.' },
  ] },
}) }));

describe('saved model evaluation table', () => {
  it('shows API values under their matching model and identifies validation versus test', () => {
    render(<MemoryRouter><Models /></MemoryRouter>);
    const table = screen.getByRole('table');
    const headers = within(table).getAllByRole('columnheader');
    expect(headers[1]).toHaveTextContent('Test MLP');
    expect(headers[1]).toHaveTextContent('Validation · 184 samples · threshold 0.50');
    expect(headers[2]).toHaveTextContent('Test · 184 samples · threshold 0.50');
    const accuracy = within(table).getByRole('rowheader', { name: 'Accuracy' }).closest('tr')!;
    expect(within(accuracy).getAllByRole('cell').map(cell => cell.textContent)).toEqual(['0.8123', '0.8400']);
    const precision = within(table).getByRole('rowheader', { name: 'Precision' }).closest('tr')!;
    expect(within(precision).getAllByRole('cell').map(cell => cell.textContent)).toEqual(['0.0000', 'Not available']);
    expect(screen.getByText(/not independent test results/)).toBeInTheDocument();
    expect(screen.getByText('Saved validation evidence for the selected artifact.')).toBeInTheDocument();
  });
});
