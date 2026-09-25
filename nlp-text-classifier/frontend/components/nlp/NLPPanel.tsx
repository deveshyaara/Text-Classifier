import ClassifierForm from "../ClassifierForm";
import ModelInfo from "../ModelInfo";
import ArchDiagram from "../ArchDiagram";
import TrainingCurves from "../TrainingCurves";

export default function NLPPanel() {
  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* ── Two-column layout ──────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
        {/* Left — Classifier */}
        <div
          className="rounded-2xl border border-white/8 p-6 sm:p-8 space-y-6"
          style={{ background: "rgba(255,255,255,0.03)" }}
        >
          <div>
            <h2 className="text-xl font-bold text-white mb-1">Analyze Text</h2>
            <p className="text-sm text-white/40">
              Enter a movie review and the model will predict its sentiment.
            </p>
          </div>
          <ClassifierForm />
        </div>

        {/* Right — Info panels */}
        <div className="space-y-8">
          {/* Performance metrics */}
          <div
            className="rounded-2xl border border-white/8 p-6"
            style={{ background: "rgba(255,255,255,0.03)" }}
          >
            <h2 className="text-lg font-bold text-white mb-4">
              Model Performance
            </h2>
            <div className="grid grid-cols-2 gap-4">
              {[
                { label: "Test Accuracy", value: "94.636%", color: "text-emerald-400" },
                { label: "Test Loss", value: "0.2265", color: "text-blue-400" },
                { label: "Parameters", value: "400,373", color: "text-violet-400" },
                { label: "Embedding Dims", value: "20", color: "text-indigo-400" },
              ].map(({ label, value, color }) => (
                <div
                  key={label}
                  className="rounded-xl border border-white/8 bg-white/3 px-4 py-3 text-center"
                >
                  <p className={`text-2xl font-black ${color}`}>{value}</p>
                  <p className="text-xs text-white/40 mt-1">{label}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Training curves */}
          <div
            className="rounded-2xl border border-white/8 p-6"
            style={{ background: "rgba(255,255,255,0.03)" }}
          >
            <TrainingCurves />
          </div>
        </div>
      </div>

      {/* ── Bottom row ─────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div
          className="rounded-2xl border border-white/8 p-6"
          style={{ background: "rgba(255,255,255,0.03)" }}
        >
          <ArchDiagram />
        </div>
        <div
          className="rounded-2xl border border-white/8 p-6"
          style={{ background: "rgba(255,255,255,0.03)" }}
        >
          <ModelInfo />
        </div>
      </div>
    </div>
  );
}
