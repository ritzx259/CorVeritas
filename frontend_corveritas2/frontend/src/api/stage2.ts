import type { AssessmentResult } from '../types';
import type { Stage2Prediction, Stage2Schema, Stage2Values } from '../components/stage2/types';
import { ApiError } from './client';

async function stage2Request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(`/api/stage2${path}`, { ...options, headers: { 'Content-Type': 'application/json' }, signal: AbortSignal.timeout(90000) }); }
  catch { throw new ApiError('Stage 2 could not be reached or timed out. Your Stage 1 result is preserved. Retry Stage 2 when ready.', 0); }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    throw new ApiError(typeof detail === 'string' ? detail : 'Review the Stage 2 inputs and retry. Your Stage 1 result is preserved.', response.status);
  }
  if (!data) throw new ApiError('Stage 2 returned an unreadable response. Your Stage 1 result is preserved.', 502);
  return data as T;
}
export const stage2Api = {
  schema: () => stage2Request<Stage2Schema>('/schema'),
  predict: (stage1: AssessmentResult, additional_features: Stage2Values) => stage2Request<Stage2Prediction>('/predict', {
    method: 'POST', body: JSON.stringify({ stage1: { model_id: stage1.response.model_id, features: stage1.features }, additional_features, explain: true }),
  }),
};
