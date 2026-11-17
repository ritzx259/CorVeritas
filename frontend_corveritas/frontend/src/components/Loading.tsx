export default function Loading({ label = "Loading model information…" }: { label?: string }) {
  return <div className="loading-state"><span className="spinner" />{label}</div>;
}
