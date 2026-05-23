type EpochPayload = {
  epoch: number;
  extras?: {
    grad_norm_mean?: number[];
    update_ratio_mean?: number[];
    dead_ratio_mean?: number[];
  };
};

function meanAbsDiff(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let sum = 0;
  for (let i = 0; i < n; i += 1) sum += Math.abs(a[i] - b[i]);
  return sum / n;
}

function firstDeadEpoch(points: EpochPayload[], layer: number, threshold: number): number | null {
  for (const p of points) {
    const val = p.extras?.dead_ratio_mean?.[layer];
    if (typeof val === "number" && val >= threshold) return p.epoch;
  }
  return null;
}

export function DiffView({ baseHistory, compareHistory }: { baseHistory: EpochPayload[]; compareHistory: EpochPayload[] }): JSX.Element {
  if (baseHistory.length === 0 || compareHistory.length === 0) {
    return (
      <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", color: "#cbd5e1" }}>
        Select two runs to compare gradients/neurons.
      </div>
    );
  }

  const maxLayers = Math.max(
    ...baseHistory.map((p) => p.extras?.grad_norm_mean?.length ?? 0),
    ...compareHistory.map((p) => p.extras?.grad_norm_mean?.length ?? 0),
  );

  const rows = Array.from({ length: maxLayers }, (_, idx) => {
    const baseGrad = baseHistory.map((p) => p.extras?.grad_norm_mean?.[idx] ?? 0);
    const compareGrad = compareHistory.map((p) => p.extras?.grad_norm_mean?.[idx] ?? 0);
    const baseUpdate = baseHistory.map((p) => p.extras?.update_ratio_mean?.[idx] ?? 0);
    const compareUpdate = compareHistory.map((p) => p.extras?.update_ratio_mean?.[idx] ?? 0);
    return {
      layer: idx + 1,
      gradDiff: meanAbsDiff(baseGrad, compareGrad),
      updateDiff: meanAbsDiff(baseUpdate, compareUpdate),
      deadBase: firstDeadEpoch(baseHistory, idx, 0.25),
      deadCompare: firstDeadEpoch(compareHistory, idx, 0.25),
    };
  });

  return (
    <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220" }}>
      <h3 style={{ marginTop: 0, marginBottom: 8, fontSize: 14 }}>Diff View</h3>
      <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ color: "#94a3b8", textAlign: "left" }}>
            <th style={{ paddingBottom: 6 }}>Layer</th>
            <th style={{ paddingBottom: 6 }}>Grad divergence</th>
            <th style={{ paddingBottom: 6 }}>Update ratio divergence</th>
            <th style={{ paddingBottom: 6 }}>Dead epoch (A)</th>
            <th style={{ paddingBottom: 6 }}>Dead epoch (B)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.layer} style={{ borderTop: "1px solid #1e293b" }}>
              <td style={{ padding: "6px 0" }}>L{row.layer}</td>
              <td>{row.gradDiff.toFixed(4)}</td>
              <td>{row.updateDiff.toFixed(4)}</td>
              <td>{row.deadBase ?? "—"}</td>
              <td>{row.deadCompare ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
