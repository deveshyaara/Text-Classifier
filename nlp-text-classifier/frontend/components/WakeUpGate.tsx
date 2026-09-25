// components/WakeUpGate.tsx
// Shows a full-screen "waking up" splash while the Render backend cold-starts.
// Once /health responds with model_loaded=true, the children are revealed.
"use client";

import { useEffect, useRef, useState } from "react";
import { API_URL } from "@/lib/api";

type Phase = "waking" | "ready" | "error";

const POLL_INTERVAL_MS  = 3000;   // poll every 3 s
const MAX_WAIT_MS       = 90000;  // give up showing spinner after 90 s
const TRANSITION_MS     = 600;    // fade-out duration

export default function WakeUpGate({ children }: { children: React.ReactNode }) {
  const [phase, setPhase]         = useState<Phase>("waking");
  const [elapsed, setElapsed]     = useState(0);   // seconds shown to user
  const [fadeOut, setFadeOut]     = useState(false);
  const startRef                  = useRef(Date.now());
  const timerRef                  = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollRef                   = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    // Elapsed-seconds counter
    timerRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
    }, 1000);

    // Health poller
    const poll = async () => {
      try {
        const res  = await fetch(`${API_URL}/health`, { cache: "no-store" });
        if (!res.ok) return;                    // non-200 → still waking
        const data = await res.json();
        if (data.model_loaded === true) {
          clearIntervals();
          setFadeOut(true);
          setTimeout(() => setPhase("ready"), TRANSITION_MS);
        }
      } catch {
        // network error → backend still sleeping, keep polling
        const waited = Date.now() - startRef.current;
        if (waited > MAX_WAIT_MS) {
          clearIntervals();
          setPhase("error");
        }
      }
    };

    poll(); // immediate first check
    pollRef.current = setInterval(poll, POLL_INTERVAL_MS);

    return () => clearIntervals();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function clearIntervals() {
    if (timerRef.current)  clearInterval(timerRef.current);
    if (pollRef.current)   clearInterval(pollRef.current);
  }

  function retry() {
    startRef.current = Date.now();
    setElapsed(0);
    setFadeOut(false);
    setPhase("waking");

    timerRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
    }, 1000);

    const poll = async () => {
      try {
        const res  = await fetch(`${API_URL}/health`, { cache: "no-store" });
        if (!res.ok) return;
        const data = await res.json();
        if (data.model_loaded === true) {
          clearIntervals();
          setFadeOut(true);
          setTimeout(() => setPhase("ready"), TRANSITION_MS);
        }
      } catch {
        const waited = Date.now() - startRef.current;
        if (waited > MAX_WAIT_MS) {
          clearIntervals();
          setPhase("error");
        }
      }
    };
    poll();
    pollRef.current = setInterval(poll, POLL_INTERVAL_MS);
  }

  // ── Already ready → render children immediately ──────────────────────────
  if (phase === "ready") return <>{children}</>;

  // ── Splash screen ─────────────────────────────────────────────────────────
  return (
    <>
      {/* Children hidden underneath so they're mounted and ready */}
      <div aria-hidden style={{ visibility: "hidden", position: "absolute", inset: 0, pointerEvents: "none" }}>
        {children}
      </div>

      {/* Overlay */}
      <div
        role="status"
        aria-live="polite"
        aria-label="Application loading"
        style={{
          opacity: fadeOut ? 0 : 1,
          transition: `opacity ${TRANSITION_MS}ms ease`,
        }}
        className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-[#0a0a12] px-6"
      >
        {/* Glow background */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden>
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                          w-[600px] h-[600px] rounded-full
                          bg-violet-600/10 blur-[120px]" />
        </div>

        <div className="relative flex flex-col items-center gap-8 max-w-sm w-full text-center">

          {/* Animated ring logo */}
          <div className="relative w-24 h-24 flex items-center justify-center">
            {/* Outer spinning ring */}
            <svg
              className="absolute inset-0 w-full h-full animate-spin"
              style={{ animationDuration: "3s" }}
              viewBox="0 0 96 96"
              fill="none"
              aria-hidden
            >
              <circle cx="48" cy="48" r="44" stroke="url(#ring-grad)" strokeWidth="3"
                      strokeLinecap="round" strokeDasharray="180 100" />
              <defs>
                <linearGradient id="ring-grad" x1="0" y1="0" x2="96" y2="96" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#7c3aed" />
                  <stop offset="1" stopColor="#6366f1" stopOpacity="0" />
                </linearGradient>
              </defs>
            </svg>
            {/* Inner pulse */}
            <div className="w-14 h-14 rounded-full bg-violet-600/20 border border-violet-500/30
                            flex items-center justify-center animate-pulse">
              {/* Brain/neural icon */}
              <svg className="w-7 h-7 text-violet-400" fill="none" viewBox="0 0 24 24"
                   stroke="currentColor" strokeWidth={1.5} aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M9.75 3.75a6 6 0 0 1 6.508 5.306A5.25 5.25 0 0 1 15 19.5H9a5.25 5.25 0 0 1-.258-10.444A6 6 0 0 1 9.75 3.75Z" />
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M9 19.5v.75m6-9.75h.008v.008H15V9.75Zm-6 0h.008v.008H9V9.75Zm3 3h.008v.008H12v-.008Zm0 3h.008v.008H12v-.008Z" />
              </svg>
            </div>
          </div>

          {/* Text */}
          {phase === "waking" && (
            <>
              <div className="space-y-2">
                <h1 className="text-2xl font-black text-white tracking-tight">
                  Waking up the{" "}
                  <span className="bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">
                    Model
                  </span>
                </h1>
                <p className="text-sm text-white/40 leading-relaxed">
                  The inference server is spinning up on Render's free tier.
                  This usually takes <span className="text-white/60 font-medium">30–60 seconds</span>.
                </p>
              </div>

              {/* Progress bar */}
              <div className="w-full space-y-2">
                <div className="h-1 w-full rounded-full bg-white/5 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-violet-600 to-indigo-500
                                transition-all duration-1000 ease-linear"
                    style={{ width: `${Math.min((elapsed / 60) * 100, 95)}%` }}
                  />
                </div>
                <p className="text-xs text-white/30 tabular-nums">
                  Waiting… {elapsed}s
                </p>
              </div>

              {/* Steps */}
              <div className="w-full rounded-xl border border-white/8 bg-white/3 p-4 space-y-2.5 text-left">
                {[
                  { done: elapsed > 5,  label: "Connecting to server" },
                  { done: elapsed > 15, label: "Loading TensorFlow runtime" },
                  { done: elapsed > 30, label: "Restoring model weights" },
                  { done: false,        label: "Running warm-up inference", active: elapsed > 30 },
                ].map(({ done, label, active }) => (
                  <div key={label} className="flex items-center gap-3">
                    <span className={`w-4 h-4 rounded-full flex-shrink-0 flex items-center justify-center
                                      text-xs font-bold transition-all duration-500 ${
                      done
                        ? "bg-emerald-500/20 border border-emerald-500/40 text-emerald-400"
                        : active
                        ? "bg-violet-500/20 border border-violet-500/40 animate-pulse"
                        : "bg-white/5 border border-white/10"
                    }`}>
                      {done ? "✓" : ""}
                    </span>
                    <span className={`text-xs transition-colors duration-300 ${
                      done ? "text-white/60" : active ? "text-white/50" : "text-white/20"
                    }`}>
                      {label}
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}

          {/* Error state */}
          {phase === "error" && (
            <>
              <div className="space-y-2">
                <h1 className="text-2xl font-black text-white tracking-tight">
                  Taking longer than{" "}
                  <span className="text-rose-400">usual</span>
                </h1>
                <p className="text-sm text-white/40 leading-relaxed">
                  The server didn't respond in 90 seconds.
                  It may be deploying or under load.
                </p>
              </div>
              <button
                id="wake-retry-btn"
                onClick={retry}
                className="px-6 py-3 rounded-xl font-semibold text-sm
                           bg-gradient-to-r from-violet-600 to-indigo-600
                           hover:from-violet-500 hover:to-indigo-500
                           text-white transition-all duration-200
                           focus:outline-none focus:ring-2 focus:ring-violet-500/60"
              >
                Try again
              </button>
            </>
          )}
        </div>
      </div>
    </>
  );
}
