type Point = { epoch: number; train_loss: number; val_loss: number; val_accuracy: number };
type Summary = { best_val_loss: number; best_val_accuracy: number; epochs: number } | null;
type Batch = { epoch: number; step: number; train_loss: number } | null;

const PAD = { top: 8, right: 10, bottom: 24, left: 42 };
const W = 320;
const H = 110;

function fmt(v: number): string {
  if (Math.abs(v) >= 1000) return v.toExponential(1);
  if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1);
  return v.toFixed(3);
}

function buildPath(values: number[], width: number, height: number): string {
  if (values.length === 0) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(1e-6, max - min);
  return values
    .map((v, i) => {
      const x = (i / Math.max(1, values.length - 1)) * width;
      const y = height - ((v - min) / range) * height;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function MiniChart({
  series,
  xLabels,
  xAxisLabel,
  yAxisLabel,
}: {
  series: { values: number[]; color: string; label?: string }[];
  xLabels?: (string | number)[];
  xAxisLabel?: string;
  yAxisLabel?: string;
}): JSX.Element {
  const pw = W - PAD.left - PAD.right;
  const ph = H - PAD.top - PAD.bottom;

  const allValues = series.flatMap((s) => s.values).filter(Number.isFinite);
  const min = allValues.length ? Math.min(...allValues) : 0;
  const max = allValues.length ? Math.max(...allValues) : 1;
  const range = Math.max(1e-9, max - min);
  const n = Math.max(...series.map((s) => s.values.length), 2);

  const toX = (i: number) => PAD.left + (i / (n - 1)) * pw;
  const toY = (v: number) => PAD.top + ph - ((v - min) / range) * ph;

  const yTicks = [min, (min + max) / 2, max];
  const xTickCount = Math.min(5, n);
  const xTicks = Array.from({ length: xTickCount }, (_, i) =>
    Math.round((i / Math.max(1, xTickCount - 1)) * (n - 1))
  );

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${W} ${H}`}
      style={{ border: "1px solid #1e293b", background: "#111827", display: "block" }}
    >
      {/* Grid lines */}
      {yTicks.map((v, i) => (
        <line key={i} x1={PAD.left} y1={toY(v)} x2={PAD.left + pw} y2={toY(v)}
          stroke="#1e293b" strokeWidth={1} strokeDasharray="3 3" />
      ))}

      {/* Axes */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + ph}
        stroke="#475569" strokeWidth={1} />
      <line x1={PAD.left} y1={PAD.top + ph} x2={PAD.left + pw} y2={PAD.top + ph}
        stroke="#475569" strokeWidth={1} />

      {/* Y ticks + labels */}
      {yTicks.map((v, i) => (
        <g key={i}>
          <line x1={PAD.left - 3} y1={toY(v)} x2={PAD.left} y2={toY(v)}
            stroke="#475569" strokeWidth={1} />
          <text x={PAD.left - 5} y={toY(v)} textAnchor="end" dominantBaseline="middle"
            fontSize={8} fill="#64748b">
            {fmt(v)}
          </text>
        </g>
      ))}

      {/* X ticks + labels */}
      {xTicks.map((idx) => (
        <g key={idx}>
          <line x1={toX(idx)} y1={PAD.top + ph} x2={toX(idx)} y2={PAD.top + ph + 3}
            stroke="#475569" strokeWidth={1} />
          <text x={toX(idx)} y={PAD.top + ph + 13} textAnchor="middle"
            fontSize={8} fill="#64748b">
            {xLabels ? xLabels[idx] ?? idx : idx + 1}
          </text>
        </g>
      ))}

      {/* X axis label */}
      {xAxisLabel && (
        <text x={PAD.left + pw / 2} y={H - 2} textAnchor="middle" fontSize={8} fill="#475569">
          {xAxisLabel}
        </text>
      )}

      {/* Y axis label */}
      {yAxisLabel && (
        <text
          x={8} y={PAD.top + ph / 2}
          textAnchor="middle" fontSize={8} fill="#475569"
          transform={`rotate(-90, 8, ${PAD.top + ph / 2})`}
        >
          {yAxisLabel}
        </text>
      )}

      {/* Data lines */}
      {series.map(({ values, color, label }) => {
        if (values.length < 2) return null;
        const d = values
          .map((v, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`)
          .join(" ");
        return <path key={label ?? color} d={d} fill="none" stroke={color}
          strokeWidth={1.5} strokeLinejoin="round" />;
      })}

      {/* Latest value dots */}
      {series.map(({ values, color, label }) =>
        values.length > 0 ? (
          <circle key={(label ?? color) + "-dot"}
            cx={toX(values.length - 1)} cy={toY(values[values.length - 1])}
            r={2.5} fill={color} />
        ) : null
      )}
    </svg>
  );
}

export function MetricsView({
  points,
  summary,
  latestBatch,
  batchLosses,
  totalEpochs,
  activeEpoch,
}: {
  points: Point[];
  summary: Summary;
  latestBatch: Batch;
  batchLosses: number[];
  totalEpochs: number;
  activeEpoch?: Point | null;
}): JSX.Element {
  const latestEpoch = points[points.length - 1];
  const displayEpoch = activeEpoch ?? latestEpoch;
  const epochXLabels = points.map((p) => p.epoch);

  return (
    <div style={{
      border: "1px solid #334155", padding: 12, background: "#0b1220",
      color: "#e2e8f0", width: "100%", boxSizing: "border-box",
      overflow: "hidden", minHeight: 420,
    }}>
      <h3 style={{ marginTop: 0 }}>Metrics</h3>
      {displayEpoch && (
        <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 6 }}>
          Viewing epoch {displayEpoch.epoch}/{totalEpochs || "?"}
        </div>
      )}

      {/* Batch loss */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ fontSize: 12, marginBottom: 4, color: "#94a3b8" }}>
          Batch loss (live)
          {latestBatch && (
            <span style={{ marginLeft: 8, color: "#f59e0b" }}>
              ep {latestBatch.epoch}/{totalEpochs} · step {latestBatch.step} · {latestBatch.train_loss.toFixed(4)}
            </span>
          )}
        </div>
        <MiniChart
          series={[{ values: batchLosses, color: "#f59e0b", label: "batch loss" }]}
          xAxisLabel="step"
          yAxisLabel="loss"
        />
      </div>

      {/* Epoch loss */}
      <div style={{ marginTop: 8 }}>
        <div style={{ fontSize: 12, marginBottom: 4, color: "#94a3b8" }}>
          Epoch loss
          {displayEpoch && (
            <span style={{ marginLeft: 8 }}>
              <span style={{ color: "#ef4444" }}>train {displayEpoch.train_loss.toFixed(4)}</span>
              <span style={{ color: "#38bdf8", marginLeft: 6 }}>val {displayEpoch.val_loss.toFixed(4)}</span>
              <span style={{ color: "#a78bfa", marginLeft: 6 }}>score {displayEpoch.val_accuracy.toFixed(4)}</span>
            </span>
          )}
        </div>
        <MiniChart
          series={[
            { values: points.map((p) => p.train_loss), color: "#ef4444", label: "train" },
            { values: points.map((p) => p.val_loss), color: "#38bdf8", label: "val" },
          ]}
          xLabels={epochXLabels}
          xAxisLabel="epoch"
          yAxisLabel="loss"
        />
        <div style={{ fontSize: 11, marginTop: 4 }}>
          <span style={{ color: "#ef4444" }}>■ Train loss</span>
          <span style={{ color: "#38bdf8", marginLeft: 10 }}>■ Val loss</span>
        </div>
      </div>

      {/* Summary */}
      <div style={{ marginTop: 12, fontSize: 13, minHeight: 74, borderTop: "1px solid #1e293b", paddingTop: 8 }}>
        {summary ? (
          <>
            <strong>Summary</strong>
            <div style={{ color: "#94a3b8", marginTop: 4 }}>
              Best val loss: <span style={{ color: "#e2e8f0" }}>{summary.best_val_loss.toFixed(4)}</span>
              <span style={{ marginLeft: 12 }}>Best val score: <span style={{ color: "#e2e8f0" }}>{summary.best_val_accuracy.toFixed(4)}</span></span>
              <span style={{ marginLeft: 12 }}>Epochs: <span style={{ color: "#e2e8f0" }}>{summary.epochs}</span></span>
            </div>
          </>
        ) : (
          <span style={{ color: "#64748b" }}>Summary: —</span>
        )}
      </div>

      {!latestBatch && !latestEpoch && !summary && (
        <div style={{ color: "#64748b", marginTop: 8 }}>No data yet. Start a run.</div>
      )}
    </div>
  );
}