// components/ClassifierForm.tsx
"use client";

import { useState } from "react";
import { predictNLP, NLPPredictResponse, APIError } from "@/lib/api";
import PredictionResult from "./PredictionResult";

const EXAMPLES = [
  "This movie was absolutely brilliant. The performances were outstanding and the direction was masterful.",
  "The film was boring and completely disappointing. I nearly fell asleep halfway through.",
  "One of the best things I have watched in years. Truly moving and unforgettable.",
  "Terrible acting, nonsensical plot, and a waste of two hours. The worst film I have seen.",
  "A beautifully crafted story with wonderful characters. I laughed and cried in equal measure.",
  "Completely unwatchable. The script was awful and the pacing was dreadful throughout.",
];

const MAX_CHARS = 10000;

export default function ClassifierForm() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<NLPPredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const res = await predictNLP(text);
      setResult(res);
    } catch (err) {
      if (err instanceof APIError) {
        setError(err.message);
      } else {
        setError("Something went wrong. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleExample = (ex: string) => {
    setText(ex);
    setResult(null);
    setError(null);
  };

  const remaining = MAX_CHARS - text.length;
  const isOverLimit = remaining < 0;

  return (
    <section aria-label="Text classifier" className="space-y-6">
      {/* Examples */}
      <div>
        <p className="text-xs uppercase tracking-widest text-white/40 mb-3">
          Try an example
        </p>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((ex, i) => (
            <button
              key={i}
              type="button"
              onClick={() => handleExample(ex)}
              className="text-xs px-3 py-1.5 rounded-lg border border-white/10 bg-white/5 
                         hover:bg-white/10 hover:border-white/20 text-white/60 hover:text-white/90
                         transition-all duration-150 text-left max-w-[200px] truncate"
              title={ex}
              aria-label={`Load example: ${ex.slice(0, 50)}…`}
            >
              {ex.slice(0, 30)}…
            </button>
          ))}
        </div>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="relative">
          <label htmlFor="review-text" className="sr-only">
            Enter movie review text
          </label>
          <textarea
            id="review-text"
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setResult(null);
              setError(null);
            }}
            placeholder="Enter a movie review… e.g. 'This film was absolutely incredible.'"
            rows={6}
            maxLength={MAX_CHARS + 1}
            className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3
                       text-white placeholder-white/25 text-sm leading-relaxed
                       focus:outline-none focus:ring-2 focus:ring-violet-500/60 focus:border-violet-500/40
                       resize-none transition-all duration-200"
            aria-describedby="char-count"
            disabled={loading}
          />
          <p
            id="char-count"
            className={`absolute bottom-3 right-3 text-xs tabular-nums transition-colors ${
              isOverLimit
                ? "text-rose-400"
                : remaining < 200
                  ? "text-amber-400"
                  : "text-white/20"
            }`}
          >
            {remaining.toLocaleString()}
          </p>
        </div>

        <button
          type="submit"
          disabled={loading || !text.trim() || isOverLimit}
          className="w-full py-3 px-6 rounded-xl font-semibold text-sm tracking-wide
                     bg-gradient-to-r from-violet-600 to-indigo-600
                     hover:from-violet-500 hover:to-indigo-500
                     disabled:opacity-40 disabled:cursor-not-allowed
                     transition-all duration-200 focus:outline-none focus:ring-2 
                     focus:ring-violet-500/60 focus:ring-offset-2 focus:ring-offset-transparent
                     text-white shadow-lg shadow-violet-900/30"
          aria-busy={loading}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg
                className="animate-spin h-4 w-4"
                viewBox="0 0 24 24"
                fill="none"
                aria-hidden="true"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v8H4z"
                />
              </svg>
              Analyzing…
            </span>
          ) : (
            "Analyze Text"
          )}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div
          role="alert"
          className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300"
        >
          {error}
        </div>
      )}

      {/* Result */}
      {result && <PredictionResult result={result} />}
    </section>
  );
}
