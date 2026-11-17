import type { Features, Health, InputSchema, ModelsResponse, Prediction } from '../types';

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); this.name = 'ApiError'; }
}
async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(`/api${path}`, { ...options, headers: { 'Content-Type': 'application/json', ...options?.headers }, signal: AbortSignal.timeout(90000) }); }
  catch { throw new ApiError('The backend could not be reached. Start FastAPI on port 8000, then try again.', 0); }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    const message = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((item: { loc?: string[]; msg: string }) => `${item.loc?.slice(-1)[0] ?? 'Input'}: ${item.msg}`).join('; ') : 'The request could not be completed. Please retry.';
    throw new ApiError(message, response.status);
  }
  return data as T;
}
export const api = {
  health: () => request<Health>('/health'),
  models: () => request<ModelsResponse>('/models'),
  schema: () => request<InputSchema>('/schema'),
  predict: (model_id: string, features: Features) => request<Prediction>('/predict', { method: 'POST', body: JSON.stringify({ model_id, features }) }),
};
