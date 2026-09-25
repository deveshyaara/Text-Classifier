"use client";

import { useState, useEffect } from "react";
import { getModels, ModelSummary } from "@/lib/api";
import ModelSelector from "./ModelSelector";
import NLPPanel from "./nlp/NLPPanel";
import ViTPanel from "./vit/ViTPanel";
import { Loader2, AlertTriangle } from "lucide-react";

export default function ModelLab() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [activeModelId, setActiveModelId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchModels() {
      try {
        const fetchedModels = await getModels();
        setModels(fetchedModels);
        // Default to the first live model
        const firstLive = fetchedModels.find(m => m.status === "live");
        if (firstLive) {
          setActiveModelId(firstLive.id);
        } else if (fetchedModels.length > 0) {
          setActiveModelId(fetchedModels[0].id);
        }
      } catch (err: any) {
        setError(err.message || "Failed to load model registry");
      } finally {
        setLoading(false);
      }
    }
    fetchModels();
  }, []);

  if (loading) {
    return (
      <div className="min-h-[400px] flex flex-col items-center justify-center text-white/50">
        <Loader2 className="w-8 h-8 animate-spin mb-4" />
        <p>Connecting to AI Registry...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-2xl mx-auto p-6 bg-red-500/10 border border-red-500/20 rounded-2xl flex items-start gap-4">
        <AlertTriangle className="w-6 h-6 text-red-400 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-red-400 font-semibold mb-1">Backend Connection Error</h3>
          <p className="text-red-300/80 text-sm">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <div className="text-center mb-10 animate-in fade-in slide-in-from-bottom-4 duration-700">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4 text-transparent bg-clip-text bg-gradient-to-r from-violet-400 via-fuchsia-400 to-indigo-400">
          AI Model Showcase
        </h1>
        <p className="text-white/60 text-lg max-w-2xl mx-auto font-light">
          A multi-model laboratory for exploring custom ML architectures.
        </p>
      </div>

      <ModelSelector 
        models={models} 
        activeModelId={activeModelId} 
        onSelect={setActiveModelId} 
      />

      <div className="mt-8 transition-all duration-500">
        {activeModelId === "nlp-sentiment" && <NLPPanel />}
        {activeModelId === "vision-transformer" && <ViTPanel />}
        {!activeModelId && (
          <div className="text-center text-white/40 py-12">
            Please select a model above to begin.
          </div>
        )}
      </div>
    </div>
  );
}
