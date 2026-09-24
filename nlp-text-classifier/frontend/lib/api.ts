// lib/api.ts — typed API client for the FastAPI backend

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface PredictRequest {
  text: string;
}

export interface PredictResponse {
  prediction: "positive" | "negative";
  confidence: number;
  probabilities: {
    positive: number;
    negative: number;
  };
  inference_time_ms: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
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

export async function predict(text: string): Promise<PredictResponse> {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    let detail = "Something went wrong. Please try again.";
    try {
      const err = await res.json();
      if (typeof err.detail === "string") {
        detail = err.detail;
      } else if (Array.isArray(err.detail) && err.detail.length > 0) {
        // Pydantic v2 returns an array of {type, loc, msg, ...} objects
        detail = err.detail.map((e: { msg?: string }) => e.msg ?? "Validation error").join("; ");
      }
    } catch {}
    throw new APIError(res.status, detail);
  }

  return res.json() as Promise<PredictResponse>;
}

export async function health(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new APIError(res.status, "Health check failed.");
  return res.json() as Promise<HealthResponse>;
}
