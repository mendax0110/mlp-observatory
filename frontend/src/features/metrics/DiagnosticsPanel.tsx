type Diagnostics = {
  grad_norms: number[];
  dead_neuron_ratio: number[];
  weight_update_norms: number[];
  activation_histograms: number[][];
} | null;

function miniBars(values: number[], color: string): JSX.Element {
  return (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(${Math.max(1, values.length)}, minmax(6px, 1fr))`, gap: 2 }}>
      {values.map((v, i) => (
        <div key={i} title={v.toFixed(4)} style={{ height: `${Math.max(4, Math.min(40, v * 10 + 4))}px`, background: color, alignSelf: "end" }} />
      ))}
    </div>
  );
}

export function DiagnosticsPanel({ diagnostics }: { diagnostics: Diagnostics }): JSX.Element {
  if (!diagnostics) {
    return (
      <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", color: "#cbd5e1", minHeight: 240 }}>
        No diagnostics yet.
      </div>
    );
  }

  return (
    <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", color: "#e2e8f0", minHeight: 240 }}>
      <h3 style={{ marginTop: 0 }}>Layer Diagnostics</h3>
      <div style={{ fontSize: 12, color: "#cbd5e1", marginBottom: 4 }}>Gradient norms</div>
      {miniBars(diagnostics.grad_norms, "#22c55e")}

      <div style={{ fontSize: 12, color: "#cbd5e1", marginTop: 8, marginBottom: 4 }}>Weight update norms</div>
      {miniBars(diagnostics.weight_update_norms, "#38bdf8")}

      <div style={{ fontSize: 12, color: "#cbd5e1", marginTop: 8, marginBottom: 4 }}>Dead neuron ratio</div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {diagnostics.dead_neuron_ratio.map((v, i) => (
          <div key={i} style={{ fontSize: 12, padding: "2px 6px", border: "1px solid #334155" }}>
            L{i + 1}: {(v * 100).toFixed(1)}%
          </div>
        ))}
      </div>

      <div style={{ fontSize: 12, color: "#cbd5e1", marginTop: 8, marginBottom: 4 }}>Activation histograms</div>
      <div style={{ display: "grid", gap: 6 }}>
        {diagnostics.activation_histograms.map((hist, i) => (
          <div key={i}>
            <div style={{ fontSize: 12, marginBottom: 2 }}>Layer {i + 1}</div>
            {miniBars(hist, "#f59e0b")}
          </div>
        ))}
      </div>
    </div>
  );
}
