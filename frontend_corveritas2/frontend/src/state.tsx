import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react';
import { api } from './api/client';
import type { AssessmentDraft, AssessmentResult, Health, InputSchema, ModelsResponse } from './types';
import { emptyDraft } from './validation/assessment';

interface ResearchState { health: Health | null; models: ModelsResponse; schema: InputSchema | null; loading: boolean; error: string | null; refresh: () => Promise<void> }
const ResearchContext = createContext<ResearchState | null>(null);
export function ResearchProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<Health | null>(null);
  const [models, setModels] = useState<ModelsResponse>({ models: [], default_model_id: null });
  const [schema, setSchema] = useState<InputSchema | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    setLoading(true); setError(null);
    const results = await Promise.allSettled([api.health(), api.models(), api.schema()]);
    if (results[0].status === 'fulfilled') setHealth(results[0].value); else setHealth(null);
    if (results[1].status === 'fulfilled') setModels(results[1].value); else setModels({ models: [], default_model_id: null });
    if (results[2].status === 'fulfilled') setSchema(results[2].value);
    const failed = results.find((result) => result.status === 'rejected');
    if (failed?.status === 'rejected') setError(failed.reason instanceof Error ? failed.reason.message : 'Backend information is unavailable.');
    setLoading(false);
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);
  return <ResearchContext.Provider value={{ health, models, schema, loading, error, refresh }}>{children}</ResearchContext.Provider>;
}
export function useResearch() { const value = useContext(ResearchContext); if (!value) throw new Error('ResearchProvider is required.'); return value; }
interface AssessmentState { draft: AssessmentDraft; setDraft: (draft: AssessmentDraft) => void; result: AssessmentResult | null; setResult: (result: AssessmentResult | null) => void; selectedModel: string; setSelectedModel: (id: string) => void; reset: () => void }
const AssessmentContext = createContext<AssessmentState | null>(null);
export function AssessmentProvider({ children }: { children: ReactNode }) {
  const [draft, setDraft] = useState<AssessmentDraft>({ ...emptyDraft });
  const [result, setResult] = useState<AssessmentResult | null>(null);
  const [selectedModel, setSelectedModel] = useState('');
  const reset = () => { setDraft({ ...emptyDraft }); setResult(null); setSelectedModel(''); };
  return <AssessmentContext.Provider value={{ draft, setDraft, result, setResult, selectedModel, setSelectedModel, reset }}>{children}</AssessmentContext.Provider>;
}
export function useAssessment() { const value = useContext(AssessmentContext); if (!value) throw new Error('AssessmentProvider is required.'); return value; }
