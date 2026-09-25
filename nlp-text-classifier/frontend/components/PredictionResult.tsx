// components/PredictionResult.tsx
"use client";

import { NLPPredictResponse } from "@/lib/api";

interface Props {
  result: NLPPredictResponse;
}

function BarFill({ value, color }: { value: number; color: string }) {
  return (
    <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-700 ease-out ${color}`}
        style={{ width: `${(value * 100).toFixed(1)}%` }}
      />
    </div>
  );
}

export default function PredictionResult({ result }: Props) {
  const isPositive = result.prediction === "positive";
  const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

  return (
    <div
      className="rounded-2xl border p-6 space-y-6 animate-fade-in"
      style={{
        background: "rgba(255,255,255,0.04)",
        borderColor: isPositive
          ? "rgba(52,211,153,0.4)"
          : "rgba(248,113,113,0.4)",
      }}
      role="region"
      aria-label="Prediction result"
    >
      {/* Label */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <p className="text-xs uppercase tracking-widest text-white/40 mb-1">
            Prediction
          </p>
          <span
            className={`text-3xl font-black uppercase tracking-wide ${
              isPositive ? "text-emerald-400" : "text-rose-400"
            }`}
          >
            {isPositive ? "😊 Positive" : "😞 Negative"}
          </span>
        </div>

        <div className="text-right">
          <p className="text-xs uppercase tracking-widest text-white/40 mb-1">
            Confidence
          </p>
          <span className="text-3xl font-black text-white">
            {pct(result.confidence)}
          </span>
        </div>
      </div>

      {/* Probability bars */}
      <div className="space-y-3">
        <p className="text-xs uppercase tracking-widest text-white/40">
          Probability Breakdown
        </p>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-emerald-400 font-medium">Positive</span>
            <span className="text-white/70 tabular-nums">
              {pct(result.probabilities.positive)}
            </span>
          </div>
          <BarFill value={result.probabilities.positive} color="bg-emerald-400" />
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-rose-400 font-medium">Negative</span>
            <span className="text-white/70 tabular-nums">
              {pct(result.probabilities.negative)}
            </span>
          </div>
          <BarFill value={result.probabilities.negative} color="bg-rose-400" />
        </div>
      </div>

      {/* Latency */}
      <p className="text-xs text-white/30 text-right">
        Inference time:{" "}
        <span className="text-white/50 font-mono">
          {result.inference_time_ms.toFixed(1)} ms
        </span>
      </p>
    </div>
  );
}
