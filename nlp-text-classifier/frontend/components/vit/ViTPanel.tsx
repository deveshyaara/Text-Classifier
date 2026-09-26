"use client";

import { useState, useRef } from "react";
import { predictViT, ViTPredictResponse } from "@/lib/api";
import { UploadCloud, Image as ImageIcon, Loader2, Info } from "lucide-react";
import ViTTrainingCurves from "./ViTTrainingCurves";

export default function ViTPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ViTPredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictViT(file);
      setResult(res);
    } catch (err: any) {
      setError(err.message || "Failed to classify image");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="bg-white/5 border border-white/10 rounded-2xl p-6 md:p-8 backdrop-blur-sm">
        
        {/* Upload Area */}
        <div 
          onClick={handleUploadClick}
          className="border-2 border-dashed border-white/20 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer hover:border-violet-500/50 hover:bg-violet-500/5 transition-all duration-300 min-h-[300px]"
        >
          {previewUrl ? (
            <img src={previewUrl} alt="Preview" className="max-h-[250px] rounded-lg shadow-lg object-contain" />
          ) : (
            <div className="flex flex-col items-center text-white/50">
              <UploadCloud className="w-12 h-12 mb-4 opacity-50" />
              <p className="font-medium text-white/80">Click to upload image</p>
              <p className="text-sm mt-2">JPG, PNG up to 5MB</p>
            </div>
          )}
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileSelect} 
            accept="image/*" 
            className="hidden" 
          />
        </div>

        {/* Action Button */}
        <div className="mt-6 flex justify-end">
          <button
            onClick={handlePredict}
            disabled={!file || loading}
            className="bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:hover:bg-violet-600 text-white font-medium px-6 py-3 rounded-xl transition-all flex items-center gap-2"
          >
            {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <ImageIcon className="w-5 h-5" />}
            <span>{loading ? "Analyzing..." : "Classify Image"}</span>
          </button>
        </div>

        {error && (
          <div className="mt-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm flex items-center gap-3">
            <Info className="w-5 h-5 flex-shrink-0" />
            <p>{error}</p>
          </div>
        )}

      </div>

      {/* Results Section */}
      {result && (
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6 md:p-8 backdrop-blur-sm space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-bold text-white flex items-center gap-2">
              <ImageIcon className="w-5 h-5 text-violet-400" />
              Prediction Result
            </h3>
            <span className="text-xs font-mono text-white/40 bg-white/5 px-2 py-1 rounded">
              {result.inference_time_ms.toFixed(1)}ms
            </span>
          </div>
          
          <div className="flex flex-col md:flex-row gap-6">
            <div className="flex-1 bg-violet-500/10 border border-violet-500/20 rounded-xl p-6 flex flex-col items-center justify-center text-center">
              <p className="text-sm text-violet-300/70 uppercase tracking-widest font-semibold mb-2">Top Class</p>
              <h4 className="text-3xl font-bold text-white capitalize">{result.prediction}</h4>
              <p className="text-violet-200 mt-2 text-lg">{(result.confidence * 100).toFixed(1)}%</p>
            </div>
            
            <div className="flex-[2] space-y-3">
              <p className="text-sm text-white/50 uppercase tracking-widest font-semibold">Top 3 Predictions</p>
              {result.top_predictions.slice(0, 3).map((pred, i) => (
                <div key={pred.class} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-white/80 capitalize">{pred.class}</span>
                    <span className="text-white/60 font-mono">{(pred.probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-violet-500 rounded-full transition-all duration-1000 ease-out"
                      style={{ width: `${pred.probability * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      
      {/* Training Curves Section */}
      <div className="mt-8">
        <ViTTrainingCurves />
      </div>
    </div>
  );
}
