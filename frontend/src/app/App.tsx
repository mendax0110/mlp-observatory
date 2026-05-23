import { useCallback, useEffect, useMemo, useState, type CSSProperties } from "react";
import { NetworkView } from "../features/network/NetworkView";
import { MetricsView } from "../features/metrics/MetricsView";
import { TrainingDynamicsPanel } from "../features/metrics/TrainingDynamicsPanel";
import { LayerTrendsPanel } from "../features/metrics/LayerTrendsPanel";
import { DistributionPanel } from "../features/metrics/DistributionPanel";
import { FeatureImportancePanel } from "../features/metrics/FeatureImportancePanel";
import { ActivationHeatmap } from "../features/network/ActivationHeatmap";
import { DiagnosticsPanel } from "../features/metrics/DiagnosticsPanel";
import { ResourceMonitor } from "../features/metrics/ResourceMonitor";
import { ForwardTraceView } from "../features/network/ForwardTraceView";
import { RecommendationsView } from "../features/metrics/RecommendationsView";
import { CompareRunsPanel } from "../features/compare/CompareRunsPanel";
import { DiffView } from "../features/compare/DiffView";
import { useRunStream } from "../lib/ws/useRunStream";
import { API_BASE } from "../lib/api/config";

type GraphPayload = {
  nodes: Array<{ id: string; layer: number; index: number }>;
  edges: Array<{ id: string; source: string; target: string }>;
};

type EpochPayload = {
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
    saturation?: {
      near_zero?: number[];
      high_mag?: number[];
    };
    weight_histograms?: number[][];
    grad_histograms?: number[][];
    train_activation_histograms?: number[][];
    val_activation_histograms?: number[][];
    confidence_calibration?: Array<{ bin: number; accuracy: number; count: number }>;
    activation_snapshot?: {
      weights?: number[];
      activations?: number[][];
      activation_stds?: number[][];
      forward_trace?: Array<{ layer: string; values: number[] }>;
    };
  };
};
type SummaryPayload = {
  best_val_loss: number;
  best_val_accuracy: number;
  epochs: number;
  recommendations?: string[];
  diagnostics?: {
    avg_dead_neuron_ratio?: number[];
    avg_grad_norm?: number[];
    avg_weight_update_norm?: number[];
  };
  feature_importance?: number[];
};

type BatchPayload = {
  epoch: number;
  step: number;
  train_loss: number;
  weights: number[];
  activations: number[][];
  activation_stds?: number[][];
  activation_histograms: number[][];
  grad_norms: number[];
  dead_neuron_ratio: number[];
  weight_update_norms: number[];
  weight_norms?: number[];
  update_ratios?: number[];
  saturation?: { near_zero: number[]; high_mag: number[] };
  forward_trace: Array<{ layer: string; values: number[] }>;
};

type SystemStats = {
  cpu_percent: number;
  ram_percent: number;
  ram_used_gb: number;
  ram_total_gb: number;
  gpu: {
    util_percent: number;
    mem_used_mb: number;
    mem_total_mb: number;
    temp_c: number;
    power_w: number;
    name: string;
  } | null;
  ts?: number;
};

type SystemStatsPoint = SystemStats & { ts: number };

type LogEntry = { ts: string; text: string };

function nowTs(): string {
  const now = new Date();
  return now.toLocaleTimeString(undefined, { hour12: false }) + "." + String(now.getMilliseconds()).padStart(3, "0");
}

type RunConfig = {
  task: "binary_classification" | "regression";
  data: {
    source: "synthetic" | "csv";
    samples: number;
    features: number;
    train_ratio: number;
    noise: number;
    seed: number;
    dataset_path: string | null;
    target_column: string | null;
  };
  model: {
    family: "mlp" | "linear";
    layers: Array<{
      units: number;
      activation: "relu" | "gelu" | "tanh" | "silu" | "leaky_relu";
      dropout: number;
      norm: "none" | "batchnorm" | "layernorm";
    }>;
    residual_every_2: boolean;
    initialization: "xavier" | "kaiming" | "orthogonal";
    preset: "custom" | "fast_baseline" | "stable_deep" | "sparse_regularized" | "high_capacity";
  };
  train: {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    weight_decay: number;
    weight_decay_decoupled: boolean;
    optimizer: "adamw" | "sgd" | "rmsprop";
    scheduler: "none" | "cosine" | "one_cycle" | "step";
    scheduler_step_size: number;
    scheduler_gamma: number;
    one_cycle_max_lr: number | null;
    momentum: number;
    gradient_clip_norm: number | null;
    early_stopping_patience: number | null;
    early_stopping_min_delta: number;
    label_smoothing: number;
    mixed_precision: boolean;
    l1_lambda: number;
    l2_lambda: number;
    input_normalization: boolean;
    seed: number;
    device: "auto" | "cuda" | "mps" | "cpu";
    trace_sample_index: number;
  };
};

type LayerConfig = RunConfig["model"]["layers"][number];

const presetLayers: Record<Exclude<RunConfig["model"]["preset"], "custom">, { model: RunConfig["model"]; train?: Partial<RunConfig["train"]> }> = {
  fast_baseline: {
    model: {
      family: "mlp",
      layers: [
        { units: 96, activation: "gelu", dropout: 0.1, norm: "none" },
        { units: 96, activation: "gelu", dropout: 0.1, norm: "none" },
        { units: 64, activation: "gelu", dropout: 0.08, norm: "none" },
      ],
      residual_every_2: false,
      initialization: "xavier",
      preset: "fast_baseline",
    },
  },
  stable_deep: {
    model: {
      family: "mlp",
      layers: [
        { units: 128, activation: "gelu", dropout: 0.12, norm: "layernorm" },
        { units: 128, activation: "gelu", dropout: 0.12, norm: "layernorm" },
        { units: 96, activation: "gelu", dropout: 0.1, norm: "layernorm" },
      ],
      residual_every_2: true,
      initialization: "xavier",
      preset: "stable_deep",
    },
  },
  sparse_regularized: {
    model: {
      family: "mlp",
      layers: [
        { units: 96, activation: "relu", dropout: 0.2, norm: "none" },
        { units: 64, activation: "relu", dropout: 0.2, norm: "none" },
        { units: 32, activation: "relu", dropout: 0.15, norm: "none" },
      ],
      residual_every_2: false,
      initialization: "kaiming",
      preset: "sparse_regularized",
    },
    train: {
      l1_lambda: 1e-5,
      l2_lambda: 1e-4,
      weight_decay: 5e-4,
    },
  },
  high_capacity: {
    model: {
      family: "mlp",
      layers: [
        { units: 192, activation: "gelu", dropout: 0.1, norm: "layernorm" },
        { units: 192, activation: "gelu", dropout: 0.1, norm: "layernorm" },
        { units: 128, activation: "gelu", dropout: 0.08, norm: "layernorm" },
        { units: 96, activation: "gelu", dropout: 0.08, norm: "layernorm" },
      ],
      residual_every_2: true,
      initialization: "orthogonal",
      preset: "high_capacity",
    },
    train: {
      gradient_clip_norm: 1.0,
      mixed_precision: true,
    },
  },
};

function cloneLayers(layers: LayerConfig[]): LayerConfig[] {
  return layers.map((layer) => ({ ...layer }));
}

const defaultConfig: RunConfig = {
  task: "binary_classification",
  data: {
    source: "synthetic",
    samples: 8192,
    features: 16,
    train_ratio: 0.8,
    noise: 0.2,
    seed: 7,
    dataset_path: null,
    target_column: null,
  },
  model: {
    family: "mlp",
    layers: [
      { units: 128, activation: "gelu", dropout: 0.12, norm: "layernorm" },
      { units: 128, activation: "gelu", dropout: 0.12, norm: "layernorm" },
      { units: 96, activation: "gelu", dropout: 0.1, norm: "layernorm" },
    ],
    residual_every_2: true,
    initialization: "xavier",
    preset: "stable_deep",
  },
  train: {
    epochs: 30,
    batch_size: 256,
    learning_rate: 0.001,
    weight_decay: 0.0001,
    weight_decay_decoupled: true,
    optimizer: "adamw",
    scheduler: "none",
    scheduler_step_size: 10,
    scheduler_gamma: 0.5,
    one_cycle_max_lr: null,
    momentum: 0.9,
    gradient_clip_norm: null,
    early_stopping_patience: null,
    early_stopping_min_delta: 0,
    label_smoothing: 0,
    mixed_precision: false,
    l1_lambda: 0,
    l2_lambda: 0,
    input_normalization: true,
    seed: 7,
    device: "auto",
    trace_sample_index: 0,
  },
};

const controlStyle: CSSProperties = {
  width: "100%",
  height: 32,
  boxSizing: "border-box",
  padding: "6px 8px",
  fontVariantNumeric: "tabular-nums",
};

const statusStyle: CSSProperties = {
  minHeight: 64,
  marginTop: 8,
  marginBottom: 8,
};

type RunListItem = { run_id: string; created_at: string; status: string; has_summary: boolean };

function formatApiError(detail: unknown, status: number): string {
  if (Array.isArray(detail)) {
    const lines = detail
      .map((d: any) => {
        const path = Array.isArray(d?.loc) ? d.loc.join(".") : "request";
        const msg = typeof d?.msg === "string" ? d.msg : "invalid value";
        return `${path}: ${msg}`;
      })
      .slice(0, 6);
    return `Validation failed (${status}): ${lines.join(" | ")}`;
  }
  if (typeof detail === "string" && detail.length > 0) {
    return `Request failed (${status}): ${detail}`;
  }
  return `Failed to start run (${status}). Check parameter ranges.`;
}

function applyPreset(config: RunConfig, preset: RunConfig["model"]["preset"]): RunConfig {
  if (preset === "custom") {
    return { ...config, model: { ...config.model, preset } };
  }
  const selected = presetLayers[preset];
  return {
    ...config,
    model: { ...config.model, ...selected.model, layers: cloneLayers(selected.model.layers), preset },
    train: { ...config.train, ...(selected.train ?? {}) },
  };
}

export function App(): JSX.Element {
  const [runId, setRunId] = useState<string | null>(null);
  const [graph, setGraph] = useState<GraphPayload>({ nodes: [], edges: [] });
  const [epochs, setEpochs] = useState<EpochPayload[]>([]);
  const [summary, setSummary] = useState<SummaryPayload | null>(null);
  const [latestBatch, setLatestBatch] = useState<BatchPayload | null>(null);
  const [edgeWeights, setEdgeWeights] = useState<number[]>([]);
  const [activations, setActivations] = useState<number[][]>([]);
  const [activationStds, setActivationStds] = useState<number[][]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [batchLosses, setBatchLosses] = useState<number[]>([]);
  const [device, setDevice] = useState<string>("-");
  const [totalEpochs, setTotalEpochs] = useState<number>(0);
  const [completedEpoch, setCompletedEpoch] = useState<number>(0);
  const [isRunFinished, setIsRunFinished] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [runConfig, setRunConfig] = useState<RunConfig>(defaultConfig);
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const [systemHistory, setSystemHistory] = useState<SystemStatsPoint[]>([]);
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>("");
  const [edgeThreshold, setEdgeThreshold] = useState<number>(0.02);
  const [collapseLayers, setCollapseLayers] = useState<boolean>(false);
  const [replayEnabled, setReplayEnabled] = useState<boolean>(false);
  const [replayEpochIndex, setReplayEpochIndex] = useState<number>(0);
  const [traceStep, setTraceStep] = useState<number>(0);
  const [isTracePlaying, setIsTracePlaying] = useState<boolean>(false);
  const [compareRunId, setCompareRunId] = useState<string>("");
  const [compareHistory, setCompareHistory] = useState<EpochPayload[]>([]);
  const [compareSystemStats, setCompareSystemStats] = useState<SystemStatsPoint[]>([]);

  const loadRuns = useCallback(async (): Promise<void> => {
    try {
      const res = await fetch(`${API_BASE}/runs`);
      if (!res.ok) return;
      const data = await res.json();
      const list = Array.isArray(data.runs) ? data.runs : [];
      setRuns(list);
      if (!runId && list.length > 0) {
        setSelectedRun(list[0].run_id);
      }
    } catch {
      // Ignore transient fetch failures.
    }
  }, [runId]);

  useEffect(() => {
    void loadRuns();
  }, [loadRuns]);

  const loadHistory = useCallback(async (id: string): Promise<void> => {
    try {
      const res = await fetch(`${API_BASE}/runs/${id}/history`);
      if (!res.ok) return;
      const payload = await res.json();
      const history = Array.isArray(payload.history) ? payload.history : [];
      setEpochs(history);
      setReplayEpochIndex(Math.max(0, history.length - 1));
    } catch {
      // ignore
    }
  }, []);

  const loadSystemStats = useCallback(async (id: string): Promise<void> => {
    try {
      const res = await fetch(`${API_BASE}/runs/${id}/system-stats`);
      if (!res.ok) return;
      const payload = await res.json();
      const stats = Array.isArray(payload.system_stats) ? payload.system_stats : [];
      const withTs = stats.map((s: SystemStats, idx: number) => ({ ...s, ts: typeof s.ts === "number" ? s.ts : idx }));
      setSystemHistory(withTs as SystemStatsPoint[]);
    } catch {
      // ignore
    }
  }, []);

  const attachRun = useCallback(async (): Promise<void> => {
    if (!selectedRun) return;
    setError(null);
    setRunId(selectedRun);
    setEpochs([]);
    setSummary(null);
    setLatestBatch(null);
    setEdgeWeights([]);
    setGraph({ nodes: [], edges: [] });
    setActivations([]);
    setActivationStds([]);
    setBatchLosses([]);
    setCompletedEpoch(0);
    setIsRunFinished(false);
    setSystemStats(null);
    setSystemHistory([]);
    setReplayEnabled(false);
    setReplayEpochIndex(0);
    setTraceStep(0);
    setIsTracePlaying(false);
    setCompareHistory([]);
    setCompareSystemStats([]);
    setLogs((prev) => [...prev, { ts: nowTs(), text: `Attached to ${selectedRun}` }]);

    try {
      const res = await fetch(`${API_BASE}/runs/${selectedRun}`);
      if (res.ok) {
        const payload = await res.json();
        setSummary(payload);
        setIsRunFinished(true);
        setCompletedEpoch(Number(payload.epochs || 0));
        await loadHistory(selectedRun);
        await loadSystemStats(selectedRun);
      }
    } catch {
      // Ignore transient fetch failures.
    }
  }, [loadHistory, loadSystemStats, selectedRun]);

  const diagnostics = useMemo(
    () =>
      latestBatch
        ? {
            grad_norms: latestBatch.grad_norms,
            dead_neuron_ratio: latestBatch.dead_neuron_ratio,
            weight_update_norms: latestBatch.weight_update_norms,
            activation_histograms: latestBatch.activation_histograms,
          }
        : null,
    [latestBatch],
  );

  const latestTrace = latestBatch?.forward_trace ?? [];

  const activeEpoch = useMemo(() => {
    if (epochs.length === 0) return null;
    const idx = replayEnabled ? Math.min(replayEpochIndex, epochs.length - 1) : epochs.length - 1;
    return epochs[idx];
  }, [epochs, replayEnabled, replayEpochIndex]);

  useEffect(() => {
    if (!replayEnabled) {
      setReplayEpochIndex(Math.max(0, epochs.length - 1));
    }
  }, [epochs.length, replayEnabled]);

  const snapshot = activeEpoch?.extras?.activation_snapshot;
  const activeEdgeWeights = replayEnabled && snapshot?.weights?.length ? snapshot.weights : edgeWeights;
  const activeActivations = replayEnabled && snapshot?.activations?.length ? snapshot.activations : activations;
  const activeActivationStds = replayEnabled && snapshot?.activation_stds?.length ? snapshot.activation_stds : activationStds;
  const activeTrace = replayEnabled && snapshot?.forward_trace?.length ? snapshot.forward_trace : latestTrace;

  const maxEdgeWeight = useMemo(() => {
    if (!activeEdgeWeights || activeEdgeWeights.length === 0) return 1;
    return Math.max(0.5, ...activeEdgeWeights.map((w) => Math.abs(w)));
  }, [activeEdgeWeights]);

  useEffect(() => {
    setEdgeThreshold((prev) => Math.min(prev, maxEdgeWeight));
  }, [maxEdgeWeight]);

  useEffect(() => {
    setTraceStep(0);
    setIsTracePlaying(false);
  }, [activeTrace.length]);

  useEffect(() => {
    if (!isTracePlaying || activeTrace.length === 0) return;
    const handle = window.setInterval(() => {
      setTraceStep((prev) => (prev + 1) % activeTrace.length);
    }, 700);
    return () => window.clearInterval(handle);
  }, [activeTrace.length, isTracePlaying]);

  const maxLayer = useMemo(() => graph.nodes.reduce((acc, n) => Math.max(acc, n.layer), 0), [graph.nodes]);

  const highlightedLayer = useMemo(() => {
    if (activeTrace.length === 0) return null;
    const step = activeTrace[Math.min(traceStep, activeTrace.length - 1)];
    if (!step) return null;
    if (step.layer === "input") return 0;
    if (step.layer.startsWith("hidden_")) {
      const idx = Number(step.layer.split("_")[1]);
      return Number.isFinite(idx) ? idx : null;
    }
    if (step.layer === "output") return maxLayer;
    return null;
  }, [activeTrace, traceStep, maxLayer]);

  const updateModel = useCallback((updater: (model: RunConfig["model"]) => RunConfig["model"]) => {
    setRunConfig((current) => ({ ...current, model: updater(current.model) }));
  }, []);

  const updateLayer = useCallback(
    (index: number, patch: Partial<LayerConfig>) => {
      updateModel((model) => ({
        ...model,
        preset: "custom",
        layers: model.layers.map((layer, layerIndex) => (layerIndex === index ? { ...layer, ...patch } : layer)),
      }));
    },
    [updateModel],
  );

  const startRun = async (): Promise<void> => {
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(runConfig),
      });
      if (!response.ok) {
        let detail: unknown = null;
        try {
          const payload = await response.json();
          detail = payload?.detail;
        } catch {
          detail = null;
        }
        throw new Error(formatApiError(detail, response.status));
      }
      const data = await response.json();
      setRunId(data.run_id);
      setEpochs([]);
      setSummary(null);
      setLatestBatch(null);
      setEdgeWeights([]);
      setGraph({ nodes: [], edges: [] });
      setActivations([]);
      setActivationStds([]);
      setBatchLosses([]);
      setCompletedEpoch(0);
      setIsRunFinished(false);
      setSystemStats(null);
      setSystemHistory([]);
      setReplayEnabled(false);
      setReplayEpochIndex(0);
      setTraceStep(0);
      setIsTracePlaying(false);
      setLogs([
        {
          ts: nowTs(),
          text: `Run ${data.run_id} started | task=${runConfig.task} source=${runConfig.data.source} family=${runConfig.model.family} preset=${runConfig.model.preset} layers=${runConfig.model.layers.length} samples=${runConfig.data.samples} features=${runConfig.data.features} epochs=${runConfig.train.epochs} batch=${runConfig.train.batch_size} lr=${runConfig.train.learning_rate} device=${runConfig.train.device}`,
        },
      ]);
      void loadRuns();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  };

  const stopRun = async (): Promise<void> => {
    if (!runId) return;
    try {
      await fetch(`${API_BASE}/runs/${runId}/stop`, { method: "POST" });
      setLogs((prev) => [...prev, { ts: nowTs(), text: "Run stop requested" }]);
      void loadRuns();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  };

  const onMessage = useCallback((msg: any) => {
    if (msg.event === "run_started") {
      setGraph(msg.payload.graph);
      setDevice(msg.payload.device);
      setTotalEpochs(msg.payload.train?.epochs ?? 0);
      setLogs((prev) => {
        const next: LogEntry[] = [
          ...prev,
          {
            ts: nowTs(),
            text: `Connected. Task: ${msg.payload.task} | Family: ${msg.payload.model_family} | Device: ${msg.payload.device} (requested: ${msg.payload.requested_device})`,
          },
        ];
        if (msg.payload.device_warning) {
          next.push({ ts: nowTs(), text: msg.payload.device_warning });
        }
        return next;
      });
    }
    if (msg.event === "system_stats") {
      const payload = msg.payload as SystemStats;
      const ts = typeof payload.ts === "number" ? payload.ts : Date.now() / 1000;
      setSystemStats(payload);
      setSystemHistory((prev) => [...prev, { ...payload, ts }].slice(-2000));
    }
    if (msg.event === "batch_update") {
      const payload = msg.payload as BatchPayload;
      setLatestBatch(payload);
      setEdgeWeights(payload.weights);
      setActivations(payload.activations ?? []);
      setActivationStds(payload.activation_stds ?? []);
      setBatchLosses((prev) => [...prev, payload.train_loss].slice(-300));
      setLogs((prev) => {
        const next = [...prev, { ts: nowTs(), text: `Epoch ${payload.epoch} step ${payload.step} loss ${payload.train_loss.toFixed(4)}` }];
        return next.slice(-260);
      });
    }
    if (msg.event === "epoch_end") {
      const p = msg.payload as EpochPayload;
      setCompletedEpoch(p.epoch);
      setEpochs((prev) => {
        const next = [...prev, p];
        if (!replayEnabled) {
          setReplayEpochIndex(Math.max(0, next.length - 1));
        }
        return next;
      });
      setLogs((prev) => {
        const next = [...prev, { ts: nowTs(), text: `Epoch ${p.epoch} complete | train=${p.train_loss.toFixed(4)} val=${p.val_loss.toFixed(4)} score=${p.val_accuracy.toFixed(4)}` }];
        return next.slice(-260);
      });
    }
    if (msg.event === "run_finished") {
      const payload = msg.payload as SummaryPayload;
      setSummary(payload);
      setIsRunFinished(true);
      setCompletedEpoch(Number(payload.epochs || 0));
      setLogs((prev) => [...prev, { ts: nowTs(), text: "Run finished" }]);
      if (runId) {
        void loadHistory(runId);
        void loadSystemStats(runId);
      }
      void loadRuns();
    }
    if (msg.event === "run_failed") {
      const message = String(msg.payload?.message ?? "Unknown training error");
      setError(`Run failed: ${message}`);
      setLogs((prev) => [...prev, { ts: nowTs(), text: `Run failed: ${message}` }]);
      void loadRuns();
    }
  }, [loadHistory, loadRuns, loadSystemStats, replayEnabled, runId]);

  useRunStream(runId, onMessage);

  useEffect(() => {
    if (!compareRunId) {
      setCompareHistory([]);
      setCompareSystemStats([]);
      return;
    }

    const loadCompare = async (): Promise<void> => {
      try {
        const historyRes = await fetch(`${API_BASE}/runs/${compareRunId}/history`);
        if (historyRes.ok) {
          const payload = await historyRes.json();
          setCompareHistory(Array.isArray(payload.history) ? payload.history : []);
        }
        const statsRes = await fetch(`${API_BASE}/runs/${compareRunId}/system-stats`);
        if (statsRes.ok) {
          const payload = await statsRes.json();
          const stats = Array.isArray(payload.system_stats) ? payload.system_stats : [];
          const withTs = stats.map((s: SystemStats, idx: number) => ({ ...s, ts: typeof s.ts === "number" ? s.ts : idx }));
          setCompareSystemStats(withTs as SystemStatsPoint[]);
        }
      } catch {
        // ignore
      }
    };

    void loadCompare();
  }, [compareRunId]);

  const isMlp = runConfig.model.family === "mlp";
  const dynamicsPoints = epochs.map((epoch) => ({
    epoch: epoch.epoch,
    train_loss: epoch.train_loss,
    val_loss: epoch.val_loss,
    val_accuracy: epoch.val_accuracy,
    extras: epoch.extras,
  }));

  const replayMax = Math.max(0, epochs.length - 1);
  const replayValue = Math.min(replayEpochIndex, replayMax);

  return (
    <div style={{ fontFamily: "Inter, sans-serif", padding: 16, background: "#020617", color: "#e2e8f0", minHeight: "100vh" }}>
      <h1 style={{ marginTop: 0 }}>MLP Observatory</h1>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 12, marginBottom: 12 }}>
        <div style={{ border: "1px solid #334155", borderRadius: 8, padding: 10, background: "#0b1220" }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Task & Data</div>
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Task</label>
          <select style={{ ...controlStyle, marginBottom: 8 }} value={runConfig.task} onChange={(e) => setRunConfig((c) => ({ ...c, task: e.target.value as RunConfig["task"] }))}>
            <option value="binary_classification">binary_classification</option>
            <option value="regression">regression</option>
          </select>
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Data source</label>
          <select style={{ ...controlStyle, marginBottom: 8 }} value={runConfig.data.source} onChange={(e) => setRunConfig((c) => ({ ...c, data: { ...c.data, source: e.target.value as RunConfig["data"]["source"] } }))}>
            <option value="synthetic">synthetic</option>
            <option value="csv">csv</option>
          </select>

          {runConfig.data.source === "synthetic" ? (
            <>
              <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Samples (min 1024)</label>
              <input style={{ ...controlStyle, marginBottom: 8 }} min={1024} type="number" value={runConfig.data.samples} onChange={(e) => setRunConfig((c) => ({ ...c, data: { ...c.data, samples: Number(e.target.value) } }))} />
              <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Features (min 4)</label>
              <input style={{ ...controlStyle, marginBottom: 8 }} min={4} type="number" value={runConfig.data.features} onChange={(e) => setRunConfig((c) => ({ ...c, data: { ...c.data, features: Number(e.target.value) } }))} />
              <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Noise</label>
              <input style={controlStyle} min={0} max={1} step={0.01} type="number" value={runConfig.data.noise} onChange={(e) => setRunConfig((c) => ({ ...c, data: { ...c.data, noise: Number(e.target.value) } }))} />
            </>
          ) : (
            <>
              <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>CSV dataset path</label>
              <input
                style={{ ...controlStyle, marginBottom: 8 }}
                type="text"
                value={runConfig.data.dataset_path ?? ""}
                onChange={(e) => setRunConfig((c) => ({ ...c, data: { ...c.data, dataset_path: e.target.value || null } }))}
                placeholder="/absolute/or/relative/path.csv"
              />
              <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Target column</label>
              <input
                style={controlStyle}
                type="text"
                value={runConfig.data.target_column ?? ""}
                onChange={(e) => setRunConfig((c) => ({ ...c, data: { ...c.data, target_column: e.target.value || null } }))}
                placeholder="target"
              />
            </>
          )}
        </div>

        <div style={{ border: "1px solid #334155", borderRadius: 8, padding: 10, background: "#0b1220" }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Model Builder</div>
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Family</label>
          <select style={{ ...controlStyle, marginBottom: 8 }} value={runConfig.model.family} onChange={(e) => setRunConfig((c) => ({ ...c, model: { ...c.model, family: e.target.value as RunConfig["model"]["family"] } }))}>
            <option value="mlp">mlp</option>
            <option value="linear">linear</option>
          </select>

          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Preset</label>
          <select style={{ ...controlStyle, marginBottom: 8 }} value={runConfig.model.preset} onChange={(e) => setRunConfig((c) => applyPreset(c, e.target.value as RunConfig["model"]["preset"]))} disabled={!isMlp}>
            <option value="custom">custom</option>
            <option value="fast_baseline">fast_baseline</option>
            <option value="stable_deep">stable_deep</option>
            <option value="sparse_regularized">sparse_regularized</option>
            <option value="high_capacity">high_capacity</option>
          </select>
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1", marginTop: 8 }}>Initialization</label>
          <select style={{ ...controlStyle, marginBottom: 8 }} value={runConfig.model.initialization} onChange={(e) => updateModel((model) => ({ ...model, initialization: e.target.value as RunConfig["model"]["initialization"], preset: "custom" }))} disabled={!isMlp}>
            <option value="xavier">xavier</option>
            <option value="kaiming">kaiming</option>
            <option value="orthogonal">orthogonal</option>
          </select>
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1", marginBottom: 6 }}>Residual every 2 layers</label>
          <input type="checkbox" checked={runConfig.model.residual_every_2} onChange={(e) => updateModel((model) => ({ ...model, residual_every_2: e.target.checked, preset: "custom" }))} disabled={!isMlp} />
          <div style={{ marginTop: 10, marginBottom: 6, fontSize: 12, color: "#cbd5e1" }}>Layers</div>
          {runConfig.model.layers.map((layer, index) => (
            <div key={index} style={{ border: "1px solid #1e293b", borderRadius: 6, padding: 8, marginBottom: 8, background: "#0f172a" }}>
              <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 6 }}>Layer {index + 1}</div>
              <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Units</label>
              <input style={{ ...controlStyle, marginBottom: 8 }} min={1} max={4096} type="number" value={layer.units} onChange={(e) => updateLayer(index, { units: Number(e.target.value) })} disabled={!isMlp} />
              <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Activation</label>
              <select style={{ ...controlStyle, marginBottom: 8 }} value={layer.activation} onChange={(e) => updateLayer(index, { activation: e.target.value as LayerConfig["activation"] })} disabled={!isMlp}>
                <option value="gelu">gelu</option>
                <option value="relu">relu</option>
                <option value="tanh">tanh</option>
                <option value="silu">silu</option>
                <option value="leaky_relu">leaky_relu</option>
              </select>
              <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Dropout</label>
              <input style={{ ...controlStyle, marginBottom: 8 }} min={0} max={0.95} step={0.01} type="number" value={layer.dropout} onChange={(e) => updateLayer(index, { dropout: Number(e.target.value) })} disabled={!isMlp} />
              <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Normalization</label>
              <select style={controlStyle} value={layer.norm} onChange={(e) => updateLayer(index, { norm: e.target.value as LayerConfig["norm"] })} disabled={!isMlp}>
                <option value="none">none</option>
                <option value="batchnorm">batchnorm</option>
                <option value="layernorm">layernorm</option>
              </select>
            </div>
          ))}
        </div>

        <div style={{ border: "1px solid #334155", borderRadius: 8, padding: 10, background: "#0b1220" }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Training</div>
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Epochs (1-500)</label>
          <input style={{ ...controlStyle, marginBottom: 8 }} min={1} max={500} type="number" value={runConfig.train.epochs} onChange={(e) => setRunConfig((c) => ({ ...c, train: { ...c.train, epochs: Number(e.target.value) } }))} />
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Batch Size (min 16)</label>
          <input style={{ ...controlStyle, marginBottom: 8 }} min={16} type="number" value={runConfig.train.batch_size} onChange={(e) => setRunConfig((c) => ({ ...c, train: { ...c.train, batch_size: Number(e.target.value) } }))} />
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Learning Rate (&gt; 0)</label>
          <input style={{ ...controlStyle, marginBottom: 8 }} min={0.000001} type="number" step="0.0001" value={runConfig.train.learning_rate} onChange={(e) => setRunConfig((c) => ({ ...c, train: { ...c.train, learning_rate: Number(e.target.value) } }))} />
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Optimizer</label>
          <select style={{ ...controlStyle, marginBottom: 8 }} value={runConfig.train.optimizer} onChange={(e) => setRunConfig((c) => ({ ...c, train: { ...c.train, optimizer: e.target.value as RunConfig["train"]["optimizer"] } }))}>
            <option value="adamw">adamw</option>
            <option value="sgd">sgd</option>
            <option value="rmsprop">rmsprop</option>
          </select>
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Scheduler</label>
          <select style={{ ...controlStyle, marginBottom: 8 }} value={runConfig.train.scheduler} onChange={(e) => setRunConfig((c) => ({ ...c, train: { ...c.train, scheduler: e.target.value as RunConfig["train"]["scheduler"] } }))}>
            <option value="none">none</option>
            <option value="cosine">cosine</option>
            <option value="one_cycle">one_cycle</option>
            <option value="step">step</option>
          </select>
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Momentum</label>
          <input style={{ ...controlStyle, marginBottom: 8 }} min={0} max={0.999} step={0.01} type="number" value={runConfig.train.momentum} onChange={(e) => setRunConfig((c) => ({ ...c, train: { ...c.train, momentum: Number(e.target.value) } }))} />
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Gradient clip norm</label>
          <input style={{ ...controlStyle, marginBottom: 8 }} min={0} step={0.1} type="number" value={runConfig.train.gradient_clip_norm ?? ""} onChange={(e) => setRunConfig((c) => ({ ...c, train: { ...c.train, gradient_clip_norm: e.target.value === "" ? null : Number(e.target.value) } }))} placeholder="optional" />
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Label smoothing</label>
          <input style={{ ...controlStyle, marginBottom: 8 }} min={0} max={0.49} step={0.01} type="number" value={runConfig.train.label_smoothing} onChange={(e) => setRunConfig((c) => ({ ...c, train: { ...c.train, label_smoothing: Number(e.target.value) } }))} />
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Early stopping patience</label>
          <input style={{ ...controlStyle, marginBottom: 8 }} min={1} type="number" value={runConfig.train.early_stopping_patience ?? ""} onChange={(e) => setRunConfig((c) => ({ ...c, train: { ...c.train, early_stopping_patience: e.target.value === "" ? null : Number(e.target.value) } }))} placeholder="optional" />
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Trace sample index</label>
          <input style={{ ...controlStyle, marginBottom: 8 }} min={0} type="number" value={runConfig.train.trace_sample_index} onChange={(e) => setRunConfig((c) => ({ ...c, train: { ...c.train, trace_sample_index: Number(e.target.value) } }))} />
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Weight decay</label>
          <input style={{ ...controlStyle, marginBottom: 8 }} min={0} step={0.0001} type="number" value={runConfig.train.weight_decay} onChange={(e) => setRunConfig((c) => ({ ...c, train: { ...c.train, weight_decay: Number(e.target.value) } }))} />
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Input normalization</label>
          <input type="checkbox" checked={runConfig.train.input_normalization} onChange={(e) => setRunConfig((c) => ({ ...c, train: { ...c.train, input_normalization: e.target.checked } }))} />
          <label style={{ display: "block", fontSize: 12, color: "#cbd5e1" }}>Device</label>
          <select style={controlStyle} value={runConfig.train.device} onChange={(e) => setRunConfig((c) => ({ ...c, train: { ...c.train, device: e.target.value as RunConfig["train"]["device"] } }))}>
            <option value="auto">auto</option>
            <option value="cuda">cuda</option>
            <option value="mps">mps</option>
            <option value="cpu">cpu</option>
          </select>
        </div>
      </div>

      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
        <select style={{ ...controlStyle, maxWidth: 360 }} value={selectedRun} onChange={(e) => setSelectedRun(e.target.value)}>
          <option value="">Select a run</option>
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_id} • {r.status}
            </option>
          ))}
        </select>
        <button onClick={() => void attachRun()} disabled={!selectedRun} style={{ background: "#0ea5e9", color: "white", border: 0, padding: "8px 12px", borderRadius: 8 }}>
          Attach
        </button>
        <button onClick={() => void loadRuns()} style={{ background: "#334155", color: "white", border: 0, padding: "8px 12px", borderRadius: 8 }}>
          Refresh runs
        </button>
      </div>

      <button onClick={() => void startRun()} style={{ background: "#2563eb", color: "white", border: 0, padding: "8px 12px", borderRadius: 8, cursor: "pointer" }}>
        Start Training Run
      </button>

      <button onClick={() => void stopRun()} disabled={!runId || isRunFinished} style={{ background: "#f87171", color: "white", border: 0, padding: "8px 12px", borderRadius: 8, cursor: runId && !isRunFinished ? "pointer" : "not-allowed", marginLeft: 8 }}>
        Stop Run
      </button>

      <div style={statusStyle}>
        <p>Run ID: {runId ?? "-"} | Device: {device}</p>
        {latestBatch && !isRunFinished && completedEpoch < latestBatch.epoch && <p>Live: epoch {latestBatch.epoch}/{totalEpochs || "?"}, step {latestBatch.step}, batch loss {latestBatch.train_loss.toFixed(4)}</p>}
        {!isRunFinished && completedEpoch > 0 && (!latestBatch || completedEpoch >= latestBatch.epoch) && <p>Live: epoch {completedEpoch}/{totalEpochs || "?"} completed</p>}
        {isRunFinished && <p>Live: epoch {completedEpoch}/{totalEpochs || completedEpoch || "?"} completed • run finished</p>}
        {error && <p style={{ color: "#f87171" }}>{error}</p>}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0,2fr) minmax(0,1fr)", gap: 16, alignItems: "start" }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", marginBottom: 10 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 12, alignItems: "center" }}>
              <div>
                <div style={{ fontSize: 12, color: "#cbd5e1", marginBottom: 4 }}>Edge threshold</div>
                <input
                  type="range"
                  min={0}
                  max={maxEdgeWeight}
                  step={0.01}
                  value={edgeThreshold}
                  onChange={(e) => setEdgeThreshold(Number(e.target.value))}
                  style={{ width: "100%" }}
                />
                <div style={{ fontSize: 11, color: "#94a3b8" }}>{edgeThreshold.toFixed(3)}</div>
              </div>
              <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "#cbd5e1" }}>
                <input type="checkbox" checked={collapseLayers} onChange={(e) => setCollapseLayers(e.target.checked)} />
                Collapse layers
              </label>
            </div>
            <div style={{ marginTop: 10 }}>
              <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "#cbd5e1" }}>
                <input type="checkbox" checked={replayEnabled} onChange={(e) => setReplayEnabled(e.target.checked)} />
                Replay mode
              </label>
              <input
                type="range"
                min={0}
                max={replayMax}
                value={replayValue}
                onChange={(e) => setReplayEpochIndex(Number(e.target.value))}
                disabled={!replayEnabled}
                style={{ width: "100%", marginTop: 4 }}
              />
              <div style={{ fontSize: 11, color: "#94a3b8" }}>
                Epoch {epochs.length > 0 ? replayValue + 1 : "-"}/{epochs.length || "-"}
              </div>
            </div>
          </div>
          <NetworkView
            nodes={graph.nodes}
            edges={graph.edges}
            edgeWeights={activeEdgeWeights}
            activationMeans={activeActivations}
            activationStds={activeActivationStds}
            edgeThreshold={edgeThreshold}
            collapseLayers={collapseLayers}
            highlightedLayer={highlightedLayer}
          />
        </div>
        <div style={{ minWidth: 0, display: "grid", gap: 12 }}>
          <MetricsView
            points={epochs}
            summary={summary}
            latestBatch={latestBatch ? { epoch: latestBatch.epoch, step: latestBatch.step, train_loss: latestBatch.train_loss } : null}
            batchLosses={batchLosses}
            totalEpochs={totalEpochs}
            activeEpoch={activeEpoch ? { epoch: activeEpoch.epoch, train_loss: activeEpoch.train_loss, val_loss: activeEpoch.val_loss, val_accuracy: activeEpoch.val_accuracy } : null}
          />
          <ResourceMonitor stats={systemStats} />
          <DiagnosticsPanel diagnostics={diagnostics} />
        </div>
      </div>

      <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <ActivationHeatmap activations={activeActivations} />
        <ForwardTraceView
          trace={activeTrace}
          activeStep={traceStep}
          onStepChange={setTraceStep}
          isPlaying={isTracePlaying}
          onTogglePlay={() => setIsTracePlaying((prev) => !prev)}
        />
      </div>

      <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <DistributionPanel epoch={activeEpoch} />
        <FeatureImportancePanel importance={summary?.feature_importance} />
      </div>

      <div style={{ marginTop: 12 }}>
        <RecommendationsView recommendations={summary?.recommendations ?? []} />
      </div>

      <div style={{ marginTop: 12 }}>
        <TrainingDynamicsPanel points={dynamicsPoints} />
      </div>

      <div style={{ marginTop: 12 }}>
        <LayerTrendsPanel points={dynamicsPoints} />
      </div>

      <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <CompareRunsPanel
          runs={runs}
          currentRunId={runId}
          compareRunId={compareRunId}
          onSelect={setCompareRunId}
          baseHistory={epochs}
          compareHistory={compareHistory}
          baseSystemStats={systemHistory}
          compareSystemStats={compareSystemStats}
        />
        <DiffView baseHistory={epochs} compareHistory={compareHistory} />
      </div>

      <div style={{ marginTop: 12, border: "1px solid #334155", padding: 10, maxHeight: 260, overflowY: "auto", background: "#0b1220", width: "100%", boxSizing: "border-box" }}>
        <strong>Training Log</strong>
        {logs.map((log, i) => (
          <div key={`${log.ts}-${i}`} style={{ fontFamily: "monospace", fontSize: 12, whiteSpace: "pre-wrap", overflowWrap: "anywhere", color: "#cbd5e1" }}>
            [{log.ts}] {log.text}
          </div>
        ))}
      </div>
    </div>
  );
}