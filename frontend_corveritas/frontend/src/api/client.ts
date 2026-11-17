import type {
  Health,
  ModelsResponse,
  PatientFeatures,
  Prediction,
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new Error("Cannot reach the local assessment service. Start the FastAPI backend and try again.");
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string"
      ? body.detail
      : Array.isArray(body.detail)
        ? body.detail.map((issue: { loc?: unknown[]; msg?: string }) => {
          const field = issue.loc?.at(-1);
          return `${typeof field === "string" ? `${field}: ` : ""}${issue.msg ?? "Invalid value."}`;
        }).join(" ")
        : "The request could not be completed.";
    throw new Error(detail);
  }
  return body as T;
}

export const api = {
  health: () => request<Health>("/health"),
  models: () => request<ModelsResponse>("/models"),
  predict: (model_id: string, features: PatientFeatures) =>
    request<Prediction>("/predict", {
      method: "POST",
      body: JSON.stringify({ model_id, features }),
    }),
};

export const apiBase = API_BASE;
