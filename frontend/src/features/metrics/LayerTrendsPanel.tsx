type EpochPoint = {
  epoch: number;
  extras?: {
    grad_norm_mean?: number[];
    update_ratio_mean?: number[];
    dead_ratio_mean?: number[];
    saturation?: { near_zero?: number[]; high_mag?: number[] };
  };
};

type Series = { label: string; values: number[]; color: string };

const PAD = { top: 8, right: 10, bottom: 22, left: 38 };
const W = 240;
const H = 90;

const palette = ["#22c55e", "#38bdf8", "#f97316", "#a78bfa", "#f59e0b", "#ef4444", "#14b8a6", "#94a3b8"];

function buildLayerSeries(points: EpochPoint[], pick: (p: EpochPoint) => number[] | undefined): Series[] {
  const maxLayers = Math.max(0, ...points.map((p) => (pick(p)?.length ?? 0)));
  return Array.from({ length: maxLayers }, (_, idx) => ({
    label: `L${idx + 1}`,
    values: points.map((p) => pick(p)?.[idx] ?? 0),
    color: palette[idx % palette.length],
  }));
}

function MultiLineChart({ title, series }: { title: string; series: Series[] }): JSX.Element {
  if (series.length === 0) {
    return (
      <div style={{ border: "1px solid #1e293b", padding: 8, background: "#0f172a", color: "#64748b" }}>
        {title}: no data yet.
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
  const yTicks = [min, (min + max) / 2, max];

  return (
    <div>
      <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 4 }}>{title}</div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ border: "1px solid #1e293b", background: "#111827", display: "block" }}>
        {yTicks.map((v, i) => (
          <line key={i} x1={PAD.left} y1={toY(v)} x2={PAD.left + pw} y2={toY(v)} stroke="#1e293b" strokeWidth={1} strokeDasharray="3 3" />
        ))}
        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + ph} stroke="#475569" strokeWidth={1} />
        <line x1={PAD.left} y1={PAD.top + ph} x2={PAD.left + pw} y2={PAD.top + ph} stroke="#475569" strokeWidth={1} />

        {series.map((s) =>
          s.values.length > 1 ? (
            <path
              key={s.label}
              d={s.values.map((v, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(" ")}
              fill="none"
              stroke={s.color}
              strokeWidth={1.4}
            />
          ) : null
        )}
      </svg>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 4, fontSize: 10 }}>
        {series.map((s) => (
          <span key={s.label} style={{ color: s.color }}>{s.label}</span>
        ))}
      </div>
    </div>
  );
}

export function LayerTrendsPanel({ points }: { points: EpochPoint[] }): JSX.Element {
  const gradSeries = buildLayerSeries(points, (p) => p.extras?.grad_norm_mean);
  const updateSeries = buildLayerSeries(points, (p) => p.extras?.update_ratio_mean);
  const deadSeries = buildLayerSeries(points, (p) => p.extras?.dead_ratio_mean);
  const nearZeroSeries = buildLayerSeries(points, (p) => p.extras?.saturation?.near_zero);
  const highMagSeries = buildLayerSeries(points, (p) => p.extras?.saturation?.high_mag);

  return (
    <div style={{ border: "1px solid #334155", padding: 8, background: "#0b1220" }}>
      <h3 style={{ marginTop: 0, marginBottom: 6, fontSize: 13 }}>Layer Trends</h3>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 6 }}>
        <MultiLineChart title="Gradient norm per layer" series={gradSeries} />
        <MultiLineChart title="Update ratio per layer" series={updateSeries} />
        <MultiLineChart title="Dead neuron ratio" series={deadSeries} />
        <MultiLineChart title="Saturation near zero" series={nearZeroSeries} />
        <MultiLineChart title="Saturation high magnitude" series={highMagSeries} />
      </div>
    </div>
  );
}
