type Suggestion = {
  message: string;
  config_patch: object | null;
};

export function RecommendationsView({
  recommendations,
  suggestions,
  onApplySuggestion,
}: {
  recommendations: string[];
  suggestions?: Suggestion[];
  onApplySuggestion?: (patch: object) => void;
}): JSX.Element {
  const hasRecs = recommendations && recommendations.length > 0;
  const hasSuggestions = suggestions && suggestions.length > 0;

  if (!hasRecs && !hasSuggestions) {
    return (
      <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", color: "#cbd5e1", minHeight: 140 }}>
        No recommendations yet.
      </div>
    );
  }

  return (
    <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", color: "#e2e8f0", minHeight: 140 }}>
      <h3 style={{ marginTop: 0 }}>Evaluation & Recommendations</h3>

      {hasRecs && (
        <ul style={{ margin: 0, paddingLeft: 18, marginBottom: hasSuggestions ? 12 : 0 }}>
          {recommendations.map((r, i) => (
            <li key={i} style={{ marginBottom: 6 }}>{r}</li>
          ))}
        </ul>
      )}

      {hasSuggestions && (
        <>
          <h4 style={{ margin: "8px 0 6px", color: "#94a3b8" }}>Suggested fixes</h4>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {suggestions.map((s, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", background: "#0f172a", border: "1px solid #1e293b", borderRadius: 6, padding: "6px 10px" }}>
                <span style={{ fontSize: 13, color: "#cbd5e1" }}>{s.message}</span>
                {s.config_patch && onApplySuggestion && (
                  <button
                    onClick={() => onApplySuggestion(s.config_patch!)}
                    style={{ marginLeft: 12, flexShrink: 0, background: "#2563eb", color: "white", border: 0, borderRadius: 6, padding: "4px 10px", cursor: "pointer", fontSize: 12 }}
                  >
                    Apply
                  </button>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}