// components/ModelInfo.tsx
// Displays verified model metadata from the original notebook

export default function ModelInfo() {
  const rows = [
    { label: "Framework", value: "TensorFlow 2" },
    { label: "Embedding", value: "TF Hub gnews-swivel-20dim" },
    { label: "Embedding dims", value: "20" },
    { label: "Task", value: "Binary sentiment classification" },
    { label: "Dataset", value: "IMDb Movie Reviews" },
    { label: "Classes", value: "Positive / Negative" },
    { label: "Optimizer", value: "Adam" },
    { label: "Loss", value: "Binary Cross-Entropy (from_logits=True)" },
    { label: "Batch size", value: "100" },
    { label: "Epochs", value: "25" },
    { label: "Total parameters", value: "400,373" },
    { label: "Test accuracy", value: "94.636%" },
    { label: "Test loss", value: "0.2265" },
  ];

  return (
    <section aria-label="Model information">
      <h2 className="text-lg font-bold text-white mb-4">About the Model</h2>
      <div className="rounded-2xl border border-white/8 bg-white/3 overflow-hidden">
        <table className="w-full text-sm" role="table">
          <tbody>
            {rows.map(({ label, value }, i) => (
              <tr
                key={label}
                className={i % 2 === 0 ? "bg-white/2" : "bg-transparent"}
              >
                <td className="px-4 py-2.5 text-white/40 font-medium w-1/2">
                  {label}
                </td>
                <td className="px-4 py-2.5 text-white/80 font-mono text-xs">
                  {value}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Overfitting callout */}
      <div className="mt-4 rounded-xl border border-amber-500/20 bg-amber-500/8 px-4 py-3 text-xs text-amber-200/70 leading-relaxed">
        <span className="font-semibold text-amber-300">⚠️ Overfitting note:</span>{" "}
        Training accuracy reaches ~100% by epoch 25 while validation accuracy
        plateaus at ~87–88% from epoch 9 onward. The model memorises the
        training set. The high test accuracy (94.6%) reflects that test and
        train data share the same IMDb partition.
      </div>
    </section>
  );
}
