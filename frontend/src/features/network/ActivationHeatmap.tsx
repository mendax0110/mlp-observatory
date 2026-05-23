type Props = {
  activations: number[][];
};

function toColor(value: number): string {
  const v = Math.max(-2, Math.min(2, value));
  if (Math.abs(v) < 0.06) {
    return "#334155";
  }
  if (v > 0) {
    const c = Math.floor(240 - (v / 2) * 170);
    return `rgb(255, ${c}, ${c})`;
  }
  const c = Math.floor(240 - (Math.abs(v) / 2) * 170);
  return `rgb(${c}, ${c}, 255)`;
}

export function ActivationHeatmap({ activations }: Props): JSX.Element {
  if (activations.length === 0) {
    return (
      <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", color: "#cbd5e1", minHeight: 220 }}>
        No activation data yet.
      </div>
    );
  }

  return (
    <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", color: "#e2e8f0", minHeight: 220 }}>
      <h3 style={{ marginTop: 0 }}>Neuron Activations (mean per hidden layer)</h3>
      <div style={{ display: "grid", gap: 6 }}>
        {activations.map((layer, layerIdx) => (
          <div key={layerIdx} style={{ display: "grid", gridTemplateColumns: "110px 1fr", alignItems: "center", gap: 8 }}>
            <div style={{ fontSize: 12, color: "#cbd5e1" }}>Layer {layerIdx + 1}</div>
            <div style={{ display: "grid", gridTemplateColumns: `repeat(${layer.length}, minmax(8px, 1fr))`, gap: 2 }}>
              {layer.map((v, neuronIdx) => (
                <div
                  key={neuronIdx}
                  title={`L${layerIdx + 1}N${neuronIdx}: ${v.toFixed(4)}`}
                  style={{ height: 12, background: toColor(v), borderRadius: 2, border: "1px solid #1e293b" }}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 8, fontSize: 12, color: "#cbd5e1" }}>
        <span style={{ color: "#ef5350" }}>■ positive</span> &nbsp;
        <span style={{ color: "#5c6bc0" }}>■ negative</span>
      </div>
    </div>
  );
}
