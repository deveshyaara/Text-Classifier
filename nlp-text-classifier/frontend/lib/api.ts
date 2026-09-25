// lib/api.ts — typed API client for the FastAPI backend

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface PredictRequest {
  text: string;
}

export interface NLPPredictResponse {
  prediction: "positive" | "negative";
  confidence: number;
  probabilities: {
    positive: number;
    negative: number;
  };
  inference_time_ms: number;
}

export interface ViTPredictResponse {
  prediction: string;
  confidence: number;
  top_predictions: { class: string; probability: number }[];
  all_probabilities: Record<string, number>;
  inference_time_ms: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

export interface ModelSummary {
  id: string;
  name: string;
  type: "text" | "image";
  status: "live" | "unavailable";
}

export class APIError extends Error {
  constructor(
    public statusCode: number,
    message: string
  ) {
    super(message);
    this.name = "APIError";
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = "Something went wrong. Please try again.";
    try {
      const err = await res.json();
      if (typeof err.detail === "string") {
        detail = err.detail;
      } else if (Array.isArray(err.detail) && err.detail.length > 0) {
        detail = err.detail.map((e: { msg?: string }) => e.msg ?? "Validation error").join("; ");
      }
    } catch {}
    throw new APIError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

// ── Models API ───────────────────────────────────────────────────────────────

export async function getModels(): Promise<ModelSummary[]> {
  const res = await fetch(`${API_URL}/models`);
  return handleResponse<ModelSummary[]>(res);
}

export async function getModelMetadata(modelId: string): Promise<any> {
  const res = await fetch(`${API_URL}/models/${modelId}`);
  return handleResponse<any>(res);
}

// ── Predict API ──────────────────────────────────────────────────────────────

export async function predictNLP(text: string): Promise<NLPPredictResponse> {
  const res = await fetch(`${API_URL}/predict/nlp-sentiment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return handleResponse<NLPPredictResponse>(res);
}

export async function predictViT(imageFile: File): Promise<ViTPredictResponse> {
  const formData = new FormData();
  formData.append("file", imageFile);

  const res = await fetch(`${API_URL}/predict/vision-transformer`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<ViTPredictResponse>(res);
}

// ── Health API ───────────────────────────────────────────────────────────────

export async function health(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new APIError(res.status, "Health check failed.");
  return res.json() as Promise<HealthResponse>;
}
