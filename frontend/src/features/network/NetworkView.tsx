import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import cytoscape, { Core, ElementDefinition } from "cytoscape";

type Props = {
  nodes: Array<{ id: string; layer: number; index: number }>;
  edges: Array<{ id: string; source: string; target: string }>;
  edgeWeights: number[];
  activationMeans: number[][];
  activationStds: number[][];
  edgeThreshold: number;
  collapseLayers: boolean;
  highlightedLayer: number | null;
};

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * Soft coral ↔ white ↔ steel-blue diverging ramp.
 * w > 0  →  warm coral   (positive weight)
 * w < 0  →  cool blue    (negative weight)
 * w ≈ 0  →  near-white   (neutral)
 */
function weightToColor(w: number): string {
  const t = clamp(w, -1.5, 1.5) / 1.5; // normalised -1 … +1
  if (t >= 0) {
    const r = 255;
    const g = Math.round(255 - t * 155);
    const b = Math.round(255 - t * 175);
    return `rgb(${r},${g},${b})`;
  }
  const abs = Math.abs(t);
  const r = Math.round(255 - abs * 175);
  const g = Math.round(255 - abs * 110);
  const b = 255;
  return `rgb(${r},${g},${b})`;
}

/**
 * Node fill derived from activation mean + std.
 * Positive mean  →  warm amber tint
 * Negative mean  →  cool teal tint
 * High std       →  more saturated
 */
function nodeColor(mean: number, std: number): string {
  const m = clamp(mean, -2, 2);
  const s = clamp(std, 0, 2);
  const sat = 0.35 + s * 0.32; // 0.35 – 0.99
  const base = 38;              // dark background-ish base

  if (m >= 0) {
    const intensity = (m / 2) * sat;
    const r = Math.round(base + intensity * 210);
    const g = Math.round(base + intensity * 130);
    const b = Math.round(base + intensity * 40);
    return `rgb(${r},${g},${b})`;
  }
  const intensity = (Math.abs(m) / 2) * sat;
  const r = Math.round(base + intensity * 20);
  const g = Math.round(base + intensity * 160);
  const b = Math.round(base + intensity * 220);
  return `rgb(${r},${g},${b})`;
}

/** Glow color — slightly more saturated version of the node fill */
function glowColor(mean: number, std: number): string {
  const m = clamp(mean, -2, 2);
  if (m >= 0) return "rgba(255,160,60,0.55)";
  return "rgba(40,180,255,0.55)";
}

const LAYER_X_GAP = 150; // horizontal distance between layers (px)
const NODE_Y_GAP = 26;   // vertical distance between nodes in same layer (px)
const CANVAS_H = 620;
const CANVAS_PAD_X = 60;

export function NetworkView({
  nodes,
  edges,
  edgeWeights,
  activationMeans,
  activationStds,
  edgeThreshold,
  collapseLayers,
  highlightedLayer,
}: Props): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [zoom, setZoom] = useState(1);

  // ─── Compute preset positions ───────────────────────────────────────────────
  const positions = useMemo(() => {
    const pos: Record<string, { x: number; y: number }> = {};
    const layers = Array.from(new Set(nodes.map((n) => n.layer))).sort((a, b) => a - b);

    layers.forEach((layer) => {
      const layerNodes = nodes.filter((n) => n.layer === layer).sort((a, b) => a.index - b.index);
      const totalH = (layerNodes.length - 1) * NODE_Y_GAP;
      layerNodes.forEach((n, i) => {
        pos[n.id] = {
          x: layer * LAYER_X_GAP + CANVAS_PAD_X,
          y: CANVAS_H / 2 - totalH / 2 + i * NODE_Y_GAP,
        };
      });
    });
    return pos;
  }, [nodes]);

  // ─── Build Cytoscape elements ───────────────────────────────────────────────
  const elements = useMemo((): ElementDefinition[] => {
    if (collapseLayers) {
      const layers = Array.from(new Set(nodes.map((n) => n.layer))).sort((a, b) => a - b);

      const collapsedNodes: ElementDefinition[] = layers.map((layer) => ({
        data: {
          id: `L${layer}`,
          label: `Layer ${layer}`,
          layer,
          collapsed: true,
          highlight: highlightedLayer === layer ? 1 : 0,
          color: highlightedLayer === layer ? "#f59e0b" : "#2d3f57",
          borderColor: highlightedLayer === layer ? "#fbbf24" : "#4a6080",
          borderWidth: highlightedLayer === layer ? 2.5 : 1.5,
          size: 42,
          glow: highlightedLayer === layer ? "rgba(245,158,11,0.5)" : "rgba(100,148,180,0.25)",
        },
        position: {
          x: layer * LAYER_X_GAP + CANVAS_PAD_X,
          y: CANVAS_H / 2,
        },
      }));

      const collapsedEdges: ElementDefinition[] = layers.slice(0, -1).map((layer, idx) => ({
        data: {
          id: `CE-${layer}`,
          source: `L${layer}`,
          target: `L${layers[idx + 1]}`,
          width: 2,
          opacity: 0.6,
          color: "#64748b",
        },
      }));
      return [...collapsedNodes, ...collapsedEdges];
    }

    // ── Nodes ──────────────────────────────────────────────────────────────
    const nodeElements: ElementDefinition[] = nodes.map((n) => {
      let mean = 0;
      let std = 0;
      if (n.layer > 0 && n.layer - 1 < activationMeans.length) {
        mean = activationMeans[n.layer - 1]?.[n.index] ?? 0;
        std = activationStds[n.layer - 1]?.[n.index] ?? 0;
      }
      const isHighlighted = highlightedLayer === n.layer;
      return {
        data: {
          id: n.id,
          label: "",
          layer: n.layer,
          color: nodeColor(mean, std),
          glow: glowColor(mean, std),
          borderColor: isHighlighted ? "#f59e0b" : "rgba(255,255,255,0.12)",
          borderWidth: isHighlighted ? 2.5 : 1,
          size: isHighlighted ? 17 : 13,
        },
        position: positions[n.id] ?? { x: 0, y: 0 },
      };
    });

    // ── Edges ──────────────────────────────────────────────────────────────
    const hasWeights = edgeWeights.length > 0;

    const edgeElements: ElementDefinition[] = edges.flatMap((e, idx) => {
      const weight = hasWeights ? edgeWeights[idx % edgeWeights.length] : 0;
      const aboveThreshold = !hasWeights || Math.abs(weight) >= edgeThreshold;
      if (!aboveThreshold) return [];

      const color = hasWeights ? weightToColor(weight) : "rgba(148,163,184,0.4)";
      const rawWidth = hasWeights ? 0.7 + Math.min(4.5, Math.abs(weight) * 3.5) : 1;
      const opacity = hasWeights ? 0.45 + Math.abs(weight) * 0.2 : 0.32;

      return [{
        data: {
          id: e.id,
          source: e.source,
          target: e.target,
          weight,
          width: rawWidth,
          opacity: Math.min(opacity, 0.85),
          color,
        },
      }];
    });

    return [...nodeElements, ...edgeElements];
  }, [
    nodes,
    edges,
    edgeWeights,
    activationMeans,
    activationStds,
    edgeThreshold,
    collapseLayers,
    highlightedLayer,
    positions,
  ]);

  const minZoom = 0.2;
  const maxZoom = 2.5;

  const fitGraph = useCallback(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.fit(undefined, 28);
    setZoom(cy.zoom());
  }, []);

  const applyZoom = useCallback((value: number) => {
    const cy = cyRef.current;
    if (!cy) return;
    const next = Math.max(minZoom, Math.min(maxZoom, value));
    cy.zoom(next);
    setZoom(next);
  }, [maxZoom, minZoom]);

  const zoomBy = useCallback(
    (delta: number) => {
      const cy = cyRef.current;
      if (!cy) return;
      applyZoom(cy.zoom() + delta);
    },
    [applyZoom],
  );

  // ─── Mount / update Cytoscape ────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return;

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      userZoomingEnabled: true,
      userPanningEnabled: true,
      boxSelectionEnabled: false,
      minZoom,
      maxZoom,
      wheelSensitivity: 0.15,
      style: [
        // ── Nodes (default — individual neurons, no label) ────────────────
        {
          selector: "node",
          style: {
            width: "data(size)",
            height: "data(size)",
            "background-color": "data(color)",
            "border-color": "data(borderColor)",
            "border-width": "data(borderWidth)",
            "border-opacity": 1,
            // shadow-* ARE valid cytoscape style props but missing from older @types/cytoscape;
            // cast the whole style object to bypass the strict check.
            ...({
              "shadow-blur": 10,
              "shadow-color": "data(glow)",
              "shadow-opacity": 0.7,
              "shadow-offset-x": 0,
              "shadow-offset-y": 0,
            } as object),
            label: "",
          },
        },
        // ── Collapsed layer nodes — show label ───────────────────────────
        {
          selector: "node[collapsed]",
          style: {
            label: "data(label)",
            "font-size": 12,
            "font-weight": 600 as unknown as undefined, // cytoscape types expect number but TS definition is narrow
            color: "#f1f5f9",
            "text-valign": "center",
            "text-halign": "center",
            "font-family": '"JetBrains Mono", "Fira Mono", monospace',
            "text-outline-width": 3,
            "text-outline-color": "#0a0f1a",
          },
        },
        // ── Edges ─────────────────────────────────────────────────────────
        {
          selector: "edge",
          style: {
            // width and opacity stored as numbers in data — cast to satisfy TS
            width: "data(width)" as unknown as number,
            opacity: "data(opacity)" as unknown as number,
            "line-color": "data(color)",
            "curve-style": "unbundled-bezier",
            "control-point-distances": [20],
            "control-point-weights": [0.5],
            "target-arrow-shape": "triangle",
            "target-arrow-color": "data(color)",
            "arrow-scale": 0.55,
            "source-endpoint": "outside-to-line",
            "target-endpoint": "outside-to-line",
          },
        },
        // ── Hover state ───────────────────────────────────────────────────
        {
          selector: "node:active",
          style: {
            "overlay-opacity": 0.12,
            "overlay-color": "#f8fafc",
          },
        },
        {
          selector: "edge:active",
          style: {
            "overlay-opacity": 0,
          },
        },
      ],
      // Use preset so our hand-computed positions are respected exactly
      layout: { name: "preset" },
    });

    cyRef.current = cy;
    const syncZoom = (): void => setZoom(cy.zoom());
    cy.on("zoom", syncZoom);

    // Fit with a bit of padding
    cy.fit(undefined, 28);
    setZoom(cy.zoom());

    return () => {
      cy.removeListener("zoom", syncZoom);
      cy.destroy();
      cyRef.current = null;
    };
  }, [elements]);

  return (
    <div style={{ position: "relative", width: "100%", height: `${CANVAS_H}px` }}>
      {/* Atmospheric background container */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          borderRadius: "14px",
          background:
            "radial-gradient(ellipse at 28% 38%, #0d1f38 0%, #080d18 60%, #060810 100%)",
          border: "1px solid rgba(148,163,184,0.10)",
          boxShadow:
            "0 8px 40px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.04)",
          overflow: "hidden",
        }}
      >
        {/* Subtle grid texture */}
        <svg
          style={{ position: "absolute", inset: 0, opacity: 0.04, pointerEvents: "none" }}
          width="100%"
          height="100%"
        >
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#94a3b8" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </div>

      {/* Cytoscape canvas — sits on top of the decorative background */}
      <div
        ref={containerRef}
        style={{
          position: "absolute",
          inset: 0,
          borderRadius: "14px",
          background: "transparent",
        }}
      />

      {/* Zoom controls */}
      <div
        style={{
          position: "absolute",
          top: 12,
          right: 14,
          display: "grid",
          gap: 6,
          padding: "8px 10px",
          borderRadius: 10,
          background: "rgba(15,23,42,0.85)",
          border: "1px solid rgba(148,163,184,0.2)",
          fontSize: 10,
          color: "#cbd5e1",
        }}
      >
        <div style={{ display: "flex", gap: 6 }}>
          <button
            onClick={() => zoomBy(-0.15)}
            style={{ background: "#1e293b", color: "#e2e8f0", border: 0, borderRadius: 6, padding: "2px 8px" }}
          >
            −
          </button>
          <button
            onClick={() => zoomBy(0.15)}
            style={{ background: "#1e293b", color: "#e2e8f0", border: 0, borderRadius: 6, padding: "2px 8px" }}
          >
            +
          </button>
          <button
            onClick={fitGraph}
            style={{ background: "#0ea5e9", color: "white", border: 0, borderRadius: 6, padding: "2px 8px" }}
          >
            Fit
          </button>
        </div>
        <input
          type="range"
          min={minZoom}
          max={maxZoom}
          step={0.05}
          value={zoom}
          onChange={(e) => applyZoom(Number(e.target.value))}
          style={{ width: 140 }}
        />
        <div style={{ textAlign: "center" }}>{zoom.toFixed(2)}x</div>
      </div>

      {/* Legend pill */}
      <div
        style={{
          position: "absolute",
          bottom: 14,
          right: 16,
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "5px 12px",
          borderRadius: 999,
          background: "rgba(15,23,42,0.75)",
          backdropFilter: "blur(8px)",
          border: "1px solid rgba(148,163,184,0.12)",
          fontSize: 10,
          fontFamily: '"JetBrains Mono", monospace',
          color: "#64748b",
          pointerEvents: "none",
          userSelect: "none",
        }}
      >
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span
            style={{
              display: "inline-block",
              width: 28,
              height: 3,
              borderRadius: 2,
              background: "linear-gradient(90deg, rgb(80,130,255) 0%, rgb(255,255,255) 50%, rgb(255,100,80) 100%)",
            }}
          />
          <span>weight</span>
        </span>
        <span style={{ color: "#334155" }}>·</span>
        <span>scroll to zoom</span>
      </div>
    </div>
  );
}