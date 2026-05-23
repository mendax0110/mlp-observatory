type TraceStep = { layer: string; values: number[] };

type Props = {
  trace: TraceStep[];
  activeStep: number;
  onStepChange: (next: number) => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
};

export function ForwardTraceView({ trace, activeStep, onStepChange, isPlaying, onTogglePlay }: Props): JSX.Element {
  if (!trace || trace.length === 0) {
    return (
      <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", color: "#cbd5e1", minHeight: 220 }}>
        No forward trace yet.
      </div>
    );
  }

  const clampedStep = Math.max(0, Math.min(activeStep, trace.length - 1));
  const current = trace[clampedStep];

  return (
    <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", color: "#e2e8f0", minHeight: 220 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <h3 style={{ marginTop: 0 }}>Forward Trace (selected sample)</h3>
        <button
          onClick={onTogglePlay}
          style={{ background: isPlaying ? "#f97316" : "#1d4ed8", color: "white", border: 0, padding: "6px 10px", borderRadius: 6 }}
        >
          {isPlaying ? "Pause" : "Play"}
        </button>
      </div>
      <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 6 }}>
        Step {clampedStep + 1}/{trace.length} · {current.layer}
      </div>
      <input
        type="range"
        min={0}
        max={Math.max(0, trace.length - 1)}
        value={clampedStep}
        onChange={(e) => onStepChange(Number(e.target.value))}
        style={{ width: "100%", marginBottom: 10 }}
      />

      <div style={{ border: "1px solid #1e293b", padding: 8, background: "#111827", marginBottom: 10 }}>
        <div style={{ fontSize: 12, marginBottom: 4 }}>{current.layer}</div>
        <div style={{ fontFamily: "monospace", fontSize: 11, overflowWrap: "anywhere" }}>
          {current.values.slice(0, 16).map((v) => v.toFixed(3)).join(", ")}
        </div>
      </div>

      <div style={{ display: "grid", gap: 6, maxHeight: 160, overflowY: "auto" }}>
        {trace.map((s, i) => (
          <div key={i} style={{ border: "1px solid #1e293b", padding: 6, background: i === clampedStep ? "#1f2937" : "#0f172a" }}>
            <div style={{ fontSize: 12, marginBottom: 4 }}>{s.layer}</div>
            <div style={{ fontFamily: "monospace", fontSize: 11, overflowWrap: "anywhere" }}>
              {s.values.slice(0, 12).map((v) => v.toFixed(3)).join(", ")}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
