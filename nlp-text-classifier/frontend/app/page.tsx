// app/page.tsx — Main page
import ClassifierForm from "@/components/ClassifierForm";
import ModelInfo from "@/components/ModelInfo";
import ArchDiagram from "@/components/ArchDiagram";
import TrainingCurves from "@/components/TrainingCurves";
import { API_URL } from "@/lib/api";

export default function Home() {
  return (
    <main className="min-h-screen px-4 py-12 sm:py-16">
      <div className="mx-auto max-w-6xl">

        {/* ── Header ─────────────────────────────────────────────────────── */}
        <header className="text-center mb-14">
          <div className="inline-flex items-center gap-2 rounded-full border border-violet-500/30 bg-violet-500/10 px-4 py-1.5 text-xs text-violet-300 mb-6 font-medium tracking-wide">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            TensorFlow + TF Hub · FastAPI · Next.js
          </div>

          <h1 className="text-4xl sm:text-5xl md:text-6xl font-black text-white tracking-tight leading-tight mb-4">
            NLP Text{" "}
            <span className="bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">
              Classifier
            </span>
          </h1>

          <p className="text-white/50 text-base sm:text-lg max-w-xl mx-auto leading-relaxed">
            Binary sentiment analysis trained on IMDb movie reviews using a
            TensorFlow Hub text embedding. Classify any review as{" "}
            <span className="text-emerald-400 font-medium">positive</span> or{" "}
            <span className="text-rose-400 font-medium">negative</span>.
          </p>

          {/* API docs link */}
          <div className="mt-6 flex justify-center gap-4 flex-wrap">
            <a
              href={`${API_URL}/docs`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-white/40 hover:text-white/70 transition-colors underline underline-offset-4"
            >
              API Docs →
            </a>
            <a
              href={`${API_URL}/redoc`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-white/40 hover:text-white/70 transition-colors underline underline-offset-4"
            >
              ReDoc →
            </a>
            <a
              href={`${API_URL}/health`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-white/40 hover:text-white/70 transition-colors underline underline-offset-4"
            >
              Health →
            </a>
          </div>
        </header>

        {/* ── Two-column layout ──────────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">

          {/* Left — Classifier */}
          <div
            className="rounded-2xl border border-white/8 p-6 sm:p-8 space-y-6"
            style={{ background: "rgba(255,255,255,0.03)" }}
          >
            <div>
              <h2 className="text-xl font-bold text-white mb-1">
                Analyze Text
              </h2>
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
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
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

        {/* ── Footer ─────────────────────────────────────────────────────── */}
        <footer className="mt-14 text-center text-xs text-white/20 space-y-1">
          <p>
            Built with TensorFlow 2 · TensorFlow Hub · FastAPI · Next.js 14
          </p>
          <p>
            Model trained on IMDb movie reviews ·{" "}
            <a
              href={`${API_URL}/docs`}
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2 hover:text-white/40 transition-colors"
            >
              View API Documentation
            </a>
          </p>
        </footer>
      </div>
    </main>
  );
}
