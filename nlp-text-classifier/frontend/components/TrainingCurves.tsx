// components/TrainingCurves.tsx
// Displays actual training history from the original notebook

"use client";

// Exact values from the notebook output
const HISTORY = {
  epochs: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
  train_acc:  [0.6269,0.7685,0.8328,0.8752,0.8970,0.9176,0.9325,0.9433,0.9534,0.9613,0.9693,0.9759,0.9815,0.9867,0.9905,0.9929,0.9955,0.9970,0.9981,0.9988,0.9991,0.9994,0.9997,0.9998,0.9999],
  val_acc:    [0.6915,0.8083,0.8211,0.8465,0.8658,0.8675,0.8607,0.8777,0.8759,0.8774,0.8751,0.8761,0.8689,0.8739,0.8723,0.8696,0.8718,0.8723,0.8687,0.8706,0.8702,0.8691,0.8682,0.8677,0.8660],
  train_loss: [0.6241,0.4671,0.3678,0.2953,0.2460,0.2054,0.1755,0.1491,0.1276,0.1094,0.0929,0.0787,0.0660,0.0553,0.0456,0.0380,0.0310,0.0251,0.0206,0.0167,0.0135,0.0110,0.0089,0.0074,0.0060],
  val_loss:   [0.5308,0.4296,0.3646,0.3261,0.3048,0.2914,0.2966,0.2923,0.2899,0.2979,0.3039,0.3136,0.3332,0.3430,0.3593,0.3772,0.3965,0.4219,0.4393,0.4632,0.4782,0.4956,0.5193,0.5377,0.5591],
};

function MiniChart({
  trainData,
  valData,
  trainColor,
  valColor,
  label,
  yMin,
  yMax,
}: {
  trainData: number[];
  valData: number[];
  trainColor: string;
  valColor: string;
  label: string;
  yMin: number;
  yMax: number;
}) {
  const W = 340, H = 120, PAD = 8;
  const n = trainData.length;
  const scaleX = (i: number) => PAD + (i / (n - 1)) * (W - PAD * 2);
  const scaleY = (v: number) => H - PAD - ((v - yMin) / (yMax - yMin)) * (H - PAD * 2);

  const pathFor = (data: number[]) =>
    data.map((v, i) => `${i === 0 ? "M" : "L"}${scaleX(i)},${scaleY(v)}`).join(" ");

  return (
    <div>
      <p className="text-xs uppercase tracking-widest text-white/40 mb-2">{label}</p>
      <div className="rounded-xl border border-white/8 bg-white/3 p-3 overflow-x-auto">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto min-w-[200px]" aria-label={label}>
          <path d={pathFor(trainData)} fill="none" stroke={trainColor} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <path d={pathFor(valData)}   fill="none" stroke={valColor}   strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" strokeDasharray="5,3" />
        </svg>
        <div className="flex gap-4 mt-2 text-xs text-white/50">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-4 h-0.5 rounded" style={{ backgroundColor: trainColor }} />
            Training
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-4 h-0.5 rounded border-b-2 border-dashed" style={{ borderColor: valColor, background: "transparent" }} />
            Validation
          </span>
        </div>
      </div>
    </div>
  );
}

export default function TrainingCurves() {
  return (
    <section aria-label="Training curves">
      <h2 className="text-lg font-bold text-white mb-4">Training History</h2>
      <div className="space-y-5">
        <MiniChart
          trainData={HISTORY.train_acc}
          valData={HISTORY.val_acc}
          trainColor="#a78bfa"
          valColor="#6ee7b7"
          label="Accuracy — Training vs Validation"
          yMin={0.6}
          yMax={1.02}
        />
        <MiniChart
          trainData={HISTORY.train_loss}
          valData={HISTORY.val_loss}
          trainColor="#f472b6"
          valColor="#fb923c"
          label="Loss — Training vs Validation"
          yMin={0}
          yMax={0.65}
        />
      </div>
      <p className="mt-3 text-xs text-white/35 leading-relaxed">
        Training loss drops to near-zero. Validation loss starts increasing
        after epoch 9 — a clear sign of overfitting. Best generalisation occurs
        around epoch 8–9.
      </p>
    </section>
  );
}
