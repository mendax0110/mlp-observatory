type EpochPoint = {
  epoch: number;
  train_loss: number;
  val_loss: number;
  val_accuracy: number;
  extras?: {
    grad_norm_mean?: number[];
    weight_norm_mean?: number[];
    update_ratio_mean?: number[];
    dead_ratio_mean?: number[];
    sharpness_proxy?: number;
    saturation?: { near_zero?: number[]; high_mag?: number[] };
    lr?: number;
  };
};

function path(values: number[], w: number, h: number): string {
  if (values.length <= 1) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(1e-9, max - min);
  return values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = h - ((v - min) / range) * h;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

const PADDING = { top: 6, right: 8, bottom: 24, left: 38 };

function MiniChart({
  label,
  data,
  color,
}: {
  label: string;
  data: number[];
  color: string;
}): JSX.Element {
  const W = 250;
  const H = 80;
  const pw = W - PADDING.left - PADDING.right;
  const ph = H - PADDING.top - PADDING.bottom;

  const min = data.length > 1 ? Math.min(...data) : 0;
  const max = data.length > 1 ? Math.max(...data) : 1;
  const range = Math.max(1e-9, max - min);

  const toX = (i: number) =>
    PADDING.left + (data.length <= 1 ? 0 : (i / (data.length - 1)) * pw);
  const toY = (v: number) =>
    PADDING.top + ph - ((v - min) / range) * ph;

  const linePath =
    data.length <= 1
      ? ""
      : data
          .map((v, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`)
          .join(" ");

  // Y axis: 3 ticks
  const yTicks = [min, (min + max) / 2, max];

  // X axis: up to 5 ticks
  const xTickCount = Math.min(5, data.length);
  const xTicks =
    data.length <= 1
      ? [0]
      : Array.from({ length: xTickCount }, (_, i) =>
          Math.round((i / (xTickCount - 1)) * (data.length - 1))
        );

  const fmt = (v: number) =>
    Math.abs(v) >= 1000
      ? v.toExponential(1)
      : Math.abs(v) < 0.01 && v !== 0
      ? v.toExponential(1)
      : v.toFixed(3);

  return (
    <div>
      <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 3 }}>{label}</div>
      <svg
        width="100%"
        viewBox={`0 0 ${W} ${H}`}
        style={{ border: "1px solid #1e293b", background: "#111827", display: "block" }}
      >
        {/* Grid lines */}
        {yTicks.map((v, i) => (
          <line
            key={i}
            x1={PADDING.left}
            y1={toY(v)}
            x2={PADDING.left + pw}
            y2={toY(v)}
            stroke="#1e293b"
            strokeWidth={1}
            strokeDasharray="3 3"
          />
        ))}

        {/* Y axis line */}
        <line
          x1={PADDING.left}
          y1={PADDING.top}
          x2={PADDING.left}
          y2={PADDING.top + ph}
          stroke="#475569"
          strokeWidth={1}
        />

        {/* X axis line */}
        <line
          x1={PADDING.left}
          y1={PADDING.top + ph}
          x2={PADDING.left + pw}
          y2={PADDING.top + ph}
          stroke="#475569"
          strokeWidth={1}
        />

        {/* Y ticks + labels */}
        {yTicks.map((v, i) => (
          <g key={i}>
            <line
              x1={PADDING.left - 3}
              y1={toY(v)}
              x2={PADDING.left}
              y2={toY(v)}
              stroke="#475569"
              strokeWidth={1}
            />
            <text
              x={PADDING.left - 5}
              y={toY(v)}
              textAnchor="end"
              dominantBaseline="middle"
              fontSize={8}
              fill="#64748b"
            >
              {fmt(v)}
            </text>
          </g>
        ))}

        {/* X ticks + labels */}
        {xTicks.map((idx) => (
          <g key={idx}>
            <line
              x1={toX(idx)}
              y1={PADDING.top + ph}
              x2={toX(idx)}
              y2={PADDING.top + ph + 3}
              stroke="#475569"
              strokeWidth={1}
            />
            <text
              x={toX(idx)}
              y={PADDING.top + ph + 13}
              textAnchor="middle"
              fontSize={8}
              fill="#64748b"
            >
              {idx + 1}
            </text>
          </g>
        ))}

        {/* Data line */}
        {linePath && (
          <path d={linePath} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
        )}

        {/* Latest value dot */}
        {data.length > 0 && (
          <circle
            cx={toX(data.length - 1)}
            cy={toY(data[data.length - 1])}
            r={2.5}
            fill={color}
          />
        )}
      </svg>
    </div>
  );
}

export function TrainingDynamicsPanel({ points }: { points: EpochPoint[] }): JSX.Element {
  const grad = points.map((p) =>
    p.extras?.grad_norm_mean?.length
      ? p.extras.grad_norm_mean.reduce((a, b) => a + b, 0) / p.extras.grad_norm_mean.length
      : 0
  );
  const weightNorm = points.map((p) =>
    p.extras?.weight_norm_mean?.length
      ? p.extras.weight_norm_mean.reduce((a, b) => a + b, 0) / p.extras.weight_norm_mean.length
      : 0
  );
  const updateRatio = points.map((p) =>
    p.extras?.update_ratio_mean?.length
      ? p.extras.update_ratio_mean.reduce((a, b) => a + b, 0) / p.extras.update_ratio_mean.length
      : 0
  );
  const sharpness = points.map((p) => Number(p.extras?.sharpness_proxy ?? 0));
  const nearZero = points.map((p) =>
    p.extras?.saturation?.near_zero?.length
      ? p.extras.saturation.near_zero.reduce((a, b) => a + b, 0) / p.extras.saturation.near_zero.length
      : 0
  );
  const highMag = points.map((p) =>
    p.extras?.saturation?.high_mag?.length
      ? p.extras.saturation.high_mag.reduce((a, b) => a + b, 0) / p.extras.saturation.high_mag.length
      : 0
  );

  const charts = [
    { label: "Gradient norm trend", data: grad, color: "#22c55e" },
    { label: "Weight norm trend", data: weightNorm, color: "#14b8a6" },
    { label: "Update ratio", data: updateRatio, color: "#38bdf8" },
    { label: "Sharpness proxy", data: sharpness, color: "#f59e0b" },
    { label: "Near-zero saturation", data: nearZero, color: "#ef4444" },
    { label: "High-mag saturation", data: highMag, color: "#a855f7" },
  ];

  return (
    <div style={{ border: "1px solid #334155", padding: 8, background: "#0b1220" }}>
      <h3 style={{ marginTop: 0, marginBottom: 6, fontSize: 13 }}>Training Dynamics</h3>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 6 }}>
        {charts.map(({ label, data, color }) => (
          <MiniChart key={label} label={label} data={data} color={color} />
        ))}
      </div>
    </div>
  );
}