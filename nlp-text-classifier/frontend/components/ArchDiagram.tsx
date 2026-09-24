// components/ArchDiagram.tsx
// Visual representation of the actual inference pipeline

export default function ArchDiagram() {
  const steps = [
    { label: "User Input", sub: "Raw text string", color: "border-violet-500/40 bg-violet-500/10 text-violet-300" },
    { label: "Next.js Frontend", sub: "POST /predict", color: "border-indigo-500/40 bg-indigo-500/10 text-indigo-300" },
    { label: "FastAPI Backend", sub: "Validation → Preprocessing", color: "border-blue-500/40 bg-blue-500/10 text-blue-300" },
    { label: "TF Hub Swivel Embedding", sub: "Text → 20-dim vector", color: "border-cyan-500/40 bg-cyan-500/10 text-cyan-300" },
    { label: "Dense(16, relu)", sub: "400,020 → 16 features", color: "border-teal-500/40 bg-teal-500/10 text-teal-300" },
    { label: "Dense(1) — Logit", sub: "Raw output, no activation", color: "border-green-500/40 bg-green-500/10 text-green-300" },
    { label: "sigmoid(logit)", sub: "Logit → probability [0, 1]", color: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300" },
    { label: "Prediction", sub: "≥ 0.5 → Positive / < 0.5 → Negative", color: "border-amber-500/40 bg-amber-500/10 text-amber-300" },
  ];

  return (
    <section aria-label="Architecture diagram">
      <h2 className="text-lg font-bold text-white mb-4">Architecture</h2>
      <div className="flex flex-col items-center gap-0">
        {steps.map((step, i) => (
          <div key={step.label} className="flex flex-col items-center w-full max-w-sm">
            <div
              className={`w-full rounded-xl border px-4 py-3 text-center ${step.color}`}
            >
              <p className="font-semibold text-sm">{step.label}</p>
              <p className="text-xs opacity-70 mt-0.5">{step.sub}</p>
            </div>
            {i < steps.length - 1 && (
              <div className="text-white/20 text-lg select-none py-1">↓</div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
