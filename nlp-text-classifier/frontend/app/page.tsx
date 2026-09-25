// app/page.tsx — Main page
import ModelLab from "@/components/ModelLab";
import { API_URL } from "@/lib/api";

export default function Home() {
  return (
    <main className="min-h-screen px-4 py-12 sm:py-16">
      <div className="mx-auto max-w-6xl">

        {/* ── API links ─────────────────────────────────────────────────── */}
        <div className="flex justify-end gap-4 mb-8">
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
        </div>

        <ModelLab />

        {/* ── Footer ─────────────────────────────────────────────────────── */}
        <footer className="mt-20 text-center text-xs text-white/20 space-y-1">
          <p>
            Built with TensorFlow 2 · TensorFlow Hub · FastAPI · Next.js 14
          </p>
        </footer>
      </div>
    </main>
  );
}
