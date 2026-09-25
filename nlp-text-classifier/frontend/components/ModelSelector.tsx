"use client";

import { ModelSummary } from "@/lib/api";
import { BrainCircuit, Image as ImageIcon } from "lucide-react";

interface Props {
  models: ModelSummary[];
  activeModelId: string | null;
  onSelect: (id: string) => void;
}

export default function ModelSelector({ models, activeModelId, onSelect }: Props) {
  if (models.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-3 mb-8 justify-center">
      {models.map((m) => {
        const isActive = m.id === activeModelId;
        const Icon = m.type === "text" ? BrainCircuit : ImageIcon;

        return (
          <button
            key={m.id}
            onClick={() => onSelect(m.id)}
            disabled={m.status !== "live"}
            className={`
              relative flex items-center gap-3 px-5 py-3 rounded-xl border text-sm font-medium transition-all duration-300
              ${
                isActive
                  ? "border-violet-500/50 bg-violet-500/10 text-violet-300 shadow-[0_0_15px_rgba(139,92,246,0.1)]"
                  : "border-white/5 bg-white/5 text-white/60 hover:bg-white/10 hover:text-white"
              }
              ${m.status !== "live" ? "opacity-50 cursor-not-allowed grayscale" : ""}
            `}
          >
            <Icon className={`w-4 h-4 ${isActive ? "text-violet-400" : "text-white/40"}`} />
            <span>{m.name}</span>
            {m.status !== "live" && (
              <span className="absolute -top-2 -right-2 text-[10px] font-bold px-2 py-0.5 rounded-full bg-red-500/20 text-red-400 border border-red-500/30">
                Unavailable
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
