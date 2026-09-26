// components/vit/ViTTrainingCurves.tsx
// Renders accuracy + loss curves from training_history_vit.json
// Fetched at runtime from /training-data/training_history_vit.json (public folder)
// Falls back gracefully if file is missing (before Colab training is done).
"use client";

import { useEffect, useState } from "react";

interface ViTHistory {
  epochs: number[];
  train_loss: number[];
  train_accuracy: number[];
  val_loss: number[];
  val_accuracy: number[];
  test_loss: number;
  test_accuracy: number;
  test_top5_accuracy: number;
}

// ── Mini SVG line chart ────────────────────────────────────────────────────────

function LineChart({
  series,
  height = 120,
}: {
  series: { label: string; values: number[]; color: string }[];
  height?: number;
}) {
  const width = 400;
  const pad = { top: 8, right: 8, bottom: 24, left: 36 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const allValues = series.flatMap((s) => s.values);
  const minV = Math.min(...allValues);
  const maxV = Math.max(...allValues);
  const range = maxV - minV || 1;

  const n = series[0]?.values.length ?? 0;

  const toX = (i: number) => pad.left + (i / Math.max(n - 1, 1)) * innerW;
  const toY = (v: number) =>
    pad.top + innerH - ((v - minV) / range) * innerH;

  const polyline = (vals: number[]) =>
    vals.map((v, i) => `${toX(i)},${toY(v)}`).join(" ");

  // Y-axis ticks
  const ticks = [minV, (minV + maxV) / 2, maxV];

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full"
      aria-hidden
    >
      {/* Grid lines */}
      {ticks.map((t, i) => (
        <g key={i}>
          <line
            x1={pad.left}
            x2={pad.left + innerW}
            y1={toY(t)}
            y2={toY(t)}
            stroke="rgba(255,255,255,0.06)"
            strokeWidth={1}
          />
          <text
            x={pad.left - 4}
            y={toY(t) + 4}
            textAnchor="end"
            fontSize={8}
            fill="rgba(255,255,255,0.3)"
          >
            {t < 1 ? `${(t * 100).toFixed(0)}%` : t.toFixed(2)}
          </text>
        </g>
      ))}

      {/* X-axis label */}
      <text
        x={pad.left + innerW / 2}
        y={height - 2}
        textAnchor="middle"
        fontSize={8}
        fill="rgba(255,255,255,0.25)"
      >
        Epoch
      </text>

      {/* Lines */}
      {series.map((s) => (
        <polyline
          key={s.label}
          points={polyline(s.values)}
          fill="none"
          stroke={s.color}
          strokeWidth={1.5}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      ))}
    </svg>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function ViTTrainingCurves() {
  const [history, setHistory] = useState<ViTHistory | null>(null);
  const [notAvailable, setNotAvailable] = useState(false);

  useEffect(() => {
    fetch("/training-data/training_history_vit.json")
      .then((r) => {
        if (!r.ok) throw new Error("not found");
        return r.json();
      })
      .then(setHistory)
      .catch(() => setNotAvailable(true));
  }, []);

  if (notAvailable) {
    return (
      <div className="rounded-2xl border border-white/8 bg-white/3 p-6 text-center space-y-2">
        <p className="text-sm font-semibold text-white/60">ViT Training Curves</p>
        <p className="text-xs text-white/30">
          Not available yet — run{" "}
          <code className="text-violet-400">training/train_vit.py</code> on Colab
          and place <code className="text-violet-400">training_history_vit.json</code>{" "}
          in <code className="text-violet-400">frontend/public/training-data/</code>.
        </p>
      </div>
    );
  }

  if (!history) {
    return (
      <div className="rounded-2xl border border-white/8 bg-white/3 p-6 animate-pulse h-48" />
    );
  }

  const bestValEpoch =
    history.val_accuracy.indexOf(Math.max(...history.val_accuracy)) + 1;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-white/80 uppercase tracking-widest">
          ViT Training Curves
        </h3>
        <span className="text-xs text-white/30">
          {history.epochs.length} epochs · best @ epoch {bestValEpoch}
        </span>
      </div>

      {/* Metrics cards */}
      <div className="grid grid-cols-3 gap-3">
        {[
          {
            label: "Test Accuracy",
            value: `${(history.test_accuracy * 100).toFixed(2)}%`,
            color: "text-emerald-400",
          },
          {
            label: "Top-5 Accuracy",
            value: `${(history.test_top5_accuracy * 100).toFixed(2)}%`,
            color: "text-violet-400",
          },
          {
            label: "Test Loss",
            value: history.test_loss.toFixed(4),
            color: "text-blue-400",
          },
        ].map(({ label, value, color }) => (
          <div
            key={label}
            className="rounded-xl border border-white/8 bg-white/3 px-3 py-2 text-center"
          >
            <p className={`text-lg font-black ${color}`}>{value}</p>
            <p className="text-[10px] text-white/40 mt-0.5">{label}</p>
          </div>
        ))}
      </div>

      {/* Accuracy chart */}
      <div>
        <div className="flex items-center gap-4 mb-2">
          <p className="text-xs text-white/50 font-semibold uppercase tracking-wider">
            Accuracy
          </p>
          <div className="flex gap-3 text-[10px]">
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 bg-violet-400 inline-block rounded" />
              <span className="text-white/40">Train</span>
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 bg-emerald-400 inline-block rounded" />
              <span className="text-white/40">Val</span>
            </span>
          </div>
        </div>
        <LineChart
          series={[
            { label: "Train", values: history.train_accuracy, color: "#a78bfa" },
            { label: "Val",   values: history.val_accuracy,   color: "#34d399" },
          ]}
        />
      </div>

      {/* Loss chart */}
      <div>
        <div className="flex items-center gap-4 mb-2">
          <p className="text-xs text-white/50 font-semibold uppercase tracking-wider">
            Loss
          </p>
          <div className="flex gap-3 text-[10px]">
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 bg-violet-400 inline-block rounded" />
              <span className="text-white/40">Train</span>
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 bg-rose-400 inline-block rounded" />
              <span className="text-white/40">Val</span>
            </span>
          </div>
        </div>
        <LineChart
          series={[
            { label: "Train", values: history.train_loss, color: "#a78bfa" },
            { label: "Val",   values: history.val_loss,   color: "#fb7185" },
          ]}
        />
      </div>
    </div>
  );
}
