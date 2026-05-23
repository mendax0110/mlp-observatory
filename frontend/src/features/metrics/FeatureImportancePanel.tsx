export function FeatureImportancePanel({ importance }: { importance: number[] | null | undefined }): JSX.Element {
  if (!importance || importance.length === 0) {
    return (
      <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", color: "#cbd5e1" }}>
        No feature importance data yet.
      </div>
    );
  }

  const max = Math.max(1e-6, ...importance);
  const shown = importance.slice(0, 24);

  return (
    <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", color: "#e2e8f0" }}>
      <h3 style={{ marginTop: 0, marginBottom: 8 }}>Feature Importance</h3>
      <div style={{ display: "grid", gap: 6 }}>
        {shown.map((v, i) => (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "70px 1fr", gap: 8, alignItems: "center" }}>
            <div style={{ fontSize: 12, color: "#94a3b8" }}>F{i + 1}</div>
            <div style={{ height: 10, background: "#1e293b", position: "relative", borderRadius: 4 }}>
              <div style={{ width: `${(v / max) * 100}%`, height: "100%", background: "#f59e0b", borderRadius: 4 }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
