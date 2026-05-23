type RunListItem = { run_id: string; created_at: string; status: string; has_summary: boolean };

type EpochPayload = { epoch: number; train_loss: number; val_loss: number; val_accuracy: number };

type SystemStatsPoint = {
  ts: number;
  cpu_percent: number;
  ram_percent: number;
  gpu: { util_percent: number } | null;
};

type Series = { label: string; values: number[]; color: string };

const PAD = { top: 8, right: 10, bottom: 22, left: 38 };
const W = 320;
const H = 110;

function LineChart({ title, series }: { title: string; series: Series[] }): JSX.Element {
  if (series.every((s) => s.values.length === 0)) {
    return (
      <div style={{ border: "1px solid #1e293b", padding: 8, background: "#0f172a", color: "#64748b" }}>
        {title}: no data.
      </div>
    );
  }

  const pw = W - PAD.left - PAD.right;
  const ph = H - PAD.top - PAD.bottom;
  const allValues = series.flatMap((s) => s.values).filter(Number.isFinite);
  const min = allValues.length ? Math.min(...allValues) : 0;
  const max = allValues.length ? Math.max(...allValues) : 1;
  const range = Math.max(1e-9, max - min);
  const n = Math.max(2, ...series.map((s) => s.values.length));
  const toX = (i: number) => PAD.left + (i / (n - 1)) * pw;
  const toY = (v: number) => PAD.top + ph - ((v - min) / range) * ph;

  return (
    <div>
      <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 4 }}>{title}</div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ border: "1px solid #1e293b", background: "#111827", display: "block" }}>
        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + ph} stroke="#475569" />
        <line x1={PAD.left} y1={PAD.top + ph} x2={PAD.left + pw} y2={PAD.top + ph} stroke="#475569" />
        {series.map((s) =>
          s.values.length > 1 ? (
            <path key={s.label} d={s.values.map((v, i) => `${i === 0 ? "M" : "L"}${toX(i)},${toY(v)}`).join(" ")} fill="none" stroke={s.color} strokeWidth={1.5} />
          ) : null
        )}
      </svg>
      <div style={{ display: "flex", gap: 10, fontSize: 10, marginTop: 4 }}>
        {series.map((s) => (
          <span key={s.label} style={{ color: s.color }}>{s.label}</span>
        ))}
      </div>
    </div>
  );
}

export function CompareRunsPanel({
  runs,
  currentRunId,
  compareRunId,
  onSelect,
  baseHistory,
  compareHistory,
  baseSystemStats,
  compareSystemStats,
}: {
  runs: RunListItem[];
  currentRunId: string | null;
  compareRunId: string;
  onSelect: (runId: string) => void;
  baseHistory: EpochPayload[];
  compareHistory: EpochPayload[];
  baseSystemStats: SystemStatsPoint[];
  compareSystemStats: SystemStatsPoint[];
}): JSX.Element {
  const compareOptions = runs.filter((r) => r.run_id !== currentRunId);

  return (
    <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220" }}>
      <h3 style={{ marginTop: 0, marginBottom: 8, fontSize: 14 }}>Run Compare</h3>
      <select style={{ width: "100%", height: 32, marginBottom: 8 }} value={compareRunId} onChange={(e) => onSelect(e.target.value)}>
        <option value="">Select run to compare</option>
        {compareOptions.map((run) => (
          <option key={run.run_id} value={run.run_id}>
            {run.run_id} · {run.status}
          </option>
        ))}
      </select>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        <LineChart
          title="Train loss"
          series={[
            { label: "current", values: baseHistory.map((p) => p.train_loss), color: "#f59e0b" },
            { label: "compare", values: compareHistory.map((p) => p.train_loss), color: "#38bdf8" },
          ]}
        />
        <LineChart
          title="Val loss"
          series={[
            { label: "current", values: baseHistory.map((p) => p.val_loss), color: "#ef4444" },
            { label: "compare", values: compareHistory.map((p) => p.val_loss), color: "#a78bfa" },
          ]}
        />
        <LineChart
          title="Val accuracy"
          series={[
            { label: "current", values: baseHistory.map((p) => p.val_accuracy), color: "#22c55e" },
            { label: "compare", values: compareHistory.map((p) => p.val_accuracy), color: "#94a3b8" },
          ]}
        />
        <LineChart
          title="CPU usage"
          series={[
            { label: "current", values: baseSystemStats.map((s) => s.cpu_percent), color: "#38bdf8" },
            { label: "compare", values: compareSystemStats.map((s) => s.cpu_percent), color: "#f97316" },
          ]}
        />
        <LineChart
          title="RAM usage"
          series={[
            { label: "current", values: baseSystemStats.map((s) => s.ram_percent), color: "#14b8a6" },
            { label: "compare", values: compareSystemStats.map((s) => s.ram_percent), color: "#f472b6" },
          ]}
        />
        <LineChart
          title="GPU util"
          series={[
            { label: "current", values: baseSystemStats.map((s) => s.gpu?.util_percent ?? 0), color: "#a78bfa" },
            { label: "compare", values: compareSystemStats.map((s) => s.gpu?.util_percent ?? 0), color: "#eab308" },
          ]}
        />
      </div>
    </div>
  );
}
