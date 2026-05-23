export function RecommendationsView({ recommendations }: { recommendations: string[] }): JSX.Element {
  if (!recommendations || recommendations.length === 0) {
    return (
      <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", color: "#cbd5e1", minHeight: 140 }}>
        No recommendations yet.
      </div>
    );
  }

  return (
    <div style={{ border: "1px solid #334155", padding: 10, background: "#0b1220", color: "#e2e8f0", minHeight: 140 }}>
      <h3 style={{ marginTop: 0 }}>Evaluation & Recommendations</h3>
      <ul style={{ margin: 0, paddingLeft: 18 }}>
        {recommendations.map((r, i) => (
          <li key={i} style={{ marginBottom: 6 }}>{r}</li>
        ))}
      </ul>
    </div>
  );
}
