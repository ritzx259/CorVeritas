import { fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App';

const { research } = vi.hoisted(() => ({
  research: {
    health: { ready: true },
    models: { default_model_id: 'random_forest', models: [
      { id: 'mlp', name: 'MLP', available: false, description: 'Neural network' },
      { id: 'cnn', name: 'CNN', available: true, description: 'Convolutional network' },
      { id: 'random_forest', name: 'Random Forest', available: true, description: 'Tree ensemble' },
    ] },
    loading: false,
    error: null as string | null,
    refresh: vi.fn(),
  },
}));
vi.mock('../state', () => ({ useResearch: () => research }));

describe('immersive homepage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(window, 'scrollTo').mockImplementation(() => {});
    research.health.ready = true;
    research.error = null;
  });

  it('keeps the main routes and live model availability accessible in the new layout', async () => {
    render(<MemoryRouter><App /></MemoryRouter>);
    expect(await screen.findByRole('heading', { name: /research.*with heart/i })).toBeInTheDocument();
    const navigation = screen.getByRole('navigation', { name: 'Main navigation' });
    expect(within(navigation).getByRole('link', { name: 'Assessment' })).toHaveAttribute('href', '/assessment');
    expect(within(navigation).getByRole('link', { name: 'Models' })).toHaveAttribute('href', '/models');
    expect(within(navigation).getByRole('link', { name: 'Methodology' })).toHaveAttribute('href', '/about');
    expect(screen.getByText('Backend connected')).toBeInTheDocument();
    expect(screen.getByText('Checked by the inference backend').closest('.stat-card')).toHaveTextContent('2 / 3 models available');
    expect(screen.getByText('MLP')).toBeInTheDocument();
    expect(screen.getByText('CNN')).toBeInTheDocument();
    expect(screen.getByText('Random Forest')).toBeInTheDocument();
    expect(screen.getByText(/predictions do not establish or exclude a diagnosis/i)).toBeInTheDocument();
  });

  it('keeps backend errors and retry available without showing a connected status', async () => {
    research.health.ready = false;
    research.error = 'Backend connection failed';
    render(<MemoryRouter><App /></MemoryRouter>);
    expect(await screen.findByText('Backend not ready')).toBeInTheDocument();
    expect(screen.queryByText('Backend connected')).not.toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('Backend connection failed');
    fireEvent.click(screen.getByRole('button', { name: /try again/i }));
    expect(research.refresh).toHaveBeenCalledOnce();
  });
});
