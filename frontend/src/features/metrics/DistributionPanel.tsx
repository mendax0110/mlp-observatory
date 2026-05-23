type CalibrationPoint = { bin: number; accuracy: number; count: number };

type EpochExtras = {
  weight_histograms?: number[][];
  grad_histograms?: number[][];
  train_activation_histograms?: number[][];
  val_activation_histograms?: number[][];
  confidence_calibration?: CalibrationPoint[];
};

type EpochPayload = {
  epoch: number;
  extras?: EpochExtras;
};

function HistogramBars({ hist, color }: { hist: number[]; color: string }): JSX.Element {
  if (!hist || hist.length === 0) {
    return <div style={{ height: 6, background: "#1e293b" }} />;
  }
  const max = Math.max(1e-6, ...hist);
  return (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(${hist.length}, minmax(6px, 1fr))`, gap: 2 }}>
      {hist.map((v, i) => (
        <div key={i} title={v.toFixed(4)} style={{ height: `${Math.max(4, (v / max) * 32)}px`, background: color }} />
      ))}
    </div>
  );
}

function HistogramGrid({ title, hists, color }: { title: string; hists: number[][]; color: string }): JSX.Element {
  if (!hists || hists.length === 0) {
    return (
      <div style={{ color: "#64748b", fontSize: 12 }}>{title}: no data.</div>
    );
  }

  return (
    <div>
      <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 6 }}>{title}</div>
      <div style={{ display: "grid", gap: 6 }}>
        {hists.map((hist, i) => (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "70px 1fr", gap: 6, alignItems: "center" }}>
            <div style={{ fontSize: 11, color: "#cbd5e1" }}>L{i + 1}</div>
            <HistogramBars hist={hist} color={color} />
          </div>
        ))}
      </div>
    </div>
  );
}

function ActivationCompare({ train, val }: { train: number[][]; val: number[][] }): JSX.Element {
  if (!train || train.length === 0) {
    return <div style={{ color: "#64748b", fontSize: 12 }}>Activation histogram: no data.</div>;
  }

  return (
    <div>
      <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 6 }}>Activation histogram (train vs val)</div>
      <div style={{ display: "grid", gap: 6 }}>
        {train.map((hist, i) => (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "70px 1fr 1fr", gap: 6, alignItems: "center" }}>
            <div style={{ fontSize: 11, color: "#cbd5e1" }}>L{i + 1}</div>
            <HistogramBars hist={hist} color="#f59e0b" />
            <HistogramBars hist={val?.[i] ?? []} color="#38bdf8" />
          </div>
        ))}
      </div>
      <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 4 }}>
        <span style={{ color: "#f59e0b" }}>■ train</span> <span style={{ marginLeft: 8, color: "#38bdf8" }}>■ val</span>
      </div>
    </div>
  );
}

function CalibrationPlot({ points }: { points: CalibrationPoint[] }): JSX.Element {
  if (!points || points.length === 0) {
    return <div style={{ color: "#64748b", fontSize: 12 }}>Calibration: no data.</div>;
  }

  const W = 260;
  const H = 120;
  const PAD = 22;
  const toX = (v: number) => PAD + v * (W - PAD * 2);
  const toY = (v: number) => H - PAD - v * (H - PAD * 2);

  const line = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${toX(p.bin).toFixed(1)},${toY(p.accuracy).toFixed(1)}`)
    .join(" ");

  return (
    <div>
      <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 6 }}>Confidence calibration</div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ border: "1px solid #1e293b", background: "#111827", display: "block", maxWidth: 300 }}>
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={PAD} stroke="#334155" strokeDasharray="4 4" />
        <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#475569" />
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#475569" />
        <path d={line} fill="none" stroke="#22c55e" strokeWidth={2} />
        {points.map((p, i) => (
          <circle key={i} cx={toX(p.bin)} cy={toY(p.accuracy)} r={3} fill="#22c55e" />
        ))}
      </svg>
    </div>
  );
}

export function DistributionPanel({ epoch }: { epoch: EpochPayload | null }): JSX.Element {
  if (!epoch || !epoch.extras) {
    return (
      <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", color: "#cbd5e1" }}>
        No distribution data yet.
      </div>
    );
  }

  const extras = epoch.extras;

  return (
    <div style={{ border: "1px solid #334155", padding: 8, background: "#0b1220" }}>
      <h3 style={{ marginTop: 0, marginBottom: 6, fontSize: 13 }}>Distributions · Epoch {epoch.epoch}</h3>
      <div style={{ display: "grid", gap: 10 }}>
        <HistogramGrid title="Weight histograms" hists={extras.weight_histograms ?? []} color="#6366f1" />
        <HistogramGrid title="Gradient histograms" hists={extras.grad_histograms ?? []} color="#14b8a6" />
        <ActivationCompare train={extras.train_activation_histograms ?? []} val={extras.val_activation_histograms ?? []} />
        <CalibrationPlot points={extras.confidence_calibration ?? []} />
      </div>
    </div>
  );
}
