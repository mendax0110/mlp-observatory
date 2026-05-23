type GpuStats = {
  util_percent: number;
  mem_used_mb: number;
  mem_total_mb: number;
  temp_c: number;
  power_w: number;
  name: string;
} | null;

type Stats = {
  cpu_percent: number;
  ram_percent: number;
  ram_used_gb: number;
  ram_total_gb: number;
  gpu: GpuStats;
} | null;

function bar(value: number, color: string): JSX.Element {
  return (
    <div style={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 6, overflow: "hidden" }}>
      <div style={{ width: `${Math.max(0, Math.min(100, value))}%`, height: 10, background: color }} />
    </div>
  );
}

export function ResourceMonitor({ stats }: { stats: Stats }): JSX.Element {
  if (!stats) {
    return (
      <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", color: "#cbd5e1", minHeight: 180 }}>
        No system stats yet.
      </div>
    );
  }

  return (
    <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", color: "#e2e8f0", minHeight: 180 }}>
      <h3 style={{ marginTop: 0 }}>Resource Monitor</h3>
      <div style={{ fontSize: 13, marginBottom: 6 }}>CPU {stats.cpu_percent.toFixed(1)}%</div>
      {bar(stats.cpu_percent, "#22c55e")}
      <div style={{ fontSize: 13, marginTop: 8, marginBottom: 6 }}>
        RAM {stats.ram_percent.toFixed(1)}% ({stats.ram_used_gb.toFixed(1)}/{stats.ram_total_gb.toFixed(1)} GB)
      </div>
      {bar(stats.ram_percent, "#38bdf8")}

      {stats.gpu && (
        <>
          <div style={{ marginTop: 10, fontSize: 13, color: "#cbd5e1" }}>{stats.gpu.name}</div>
          <div style={{ fontSize: 13, marginBottom: 6 }}>GPU {stats.gpu.util_percent.toFixed(1)}%</div>
          {bar(stats.gpu.util_percent, "#f59e0b")}
          <div style={{ fontSize: 12, marginTop: 6 }}>
            VRAM {(stats.gpu.mem_used_mb / 1024).toFixed(1)}/{(stats.gpu.mem_total_mb / 1024).toFixed(1)} GB | Temp {stats.gpu.temp_c.toFixed(0)}°C | Power {stats.gpu.power_w != null ? stats.gpu.power_w.toFixed(0) : "—"}W
          </div>
        </>
      )}
    </div>
  );
}
