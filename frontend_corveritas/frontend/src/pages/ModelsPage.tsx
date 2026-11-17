import { Activity, CheckCircle2, CircleHelp, CircleX, ExternalLink } from "lucide-react";
import { useEffect, useState } from "react";
import { api, apiBase } from "../api/client";
import Loading from "../components/Loading";
import SafetyNotice from "../components/SafetyNotice";
import type { ModelsResponse } from "../types";

const metricLabels: Record<string, string> = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1: "F1",
  roc_auc: "ROC-AUC",
  average_precision: "Average precision",
};

export default function ModelsPage() {
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [error, setError] = useState("");
  useEffect(() => { api.models().then(setData).catch((reason: Error) => setError(reason.message)); }, []);
  if (!data && !error) return <Loading />;

  return (
    <>
      <div className="page-title-row"><div><span className="section-kicker">ARTIFACTS · AVAILABILITY · EVALUATION</span><h1>Model registry</h1><p>Only models that loaded successfully are offered for assessment. “Stage” names refer to saved artifacts, not disease subtypes.</p></div><span className="registry-badge"><Activity size={15} /> {data?.models.filter((model) => model.available).length ?? 0} available</span></div>
      <SafetyNotice compact />
      {error && <div className="error-banner" role="alert">Model registry unavailable: {error}</div>}
      {data?.models.map((model) => (
        <article className="model-card" key={model.id}>
          <div className="model-card-head">
            <div className="model-symbol"><Activity size={20} /></div>
            <div className="model-title"><div className="model-title-line"><h2>{model.name}</h2>{model.is_default && <span className="default-pill">Configured default</span>}</div><p>{model.architecture}</p></div>
            <div className={`availability-pill ${model.available ? "is-available" : "is-unavailable"}`}>{model.available ? <CheckCircle2 size={15} /> : <CircleX size={15} />}{model.available ? "Loaded" : "Unavailable"}</div>
          </div>
          <p className="model-description">{model.description}</p>
          {!model.available && <div className="model-error"><CircleHelp size={16} /><span>{model.error ?? "Artifact files are missing or could not be loaded."}</span></div>}
          <div className="model-detail-grid">
            <div><span>Task</span><strong>Binary classification only</strong></div>
            <div><span>Inputs</span><strong>13 UCI clinical features</strong></div>
            <div><span>Threshold</span><strong>{model.threshold.toFixed(2)}</strong></div>
            <div><span>Local explanation</span><strong>{model.id === "random_forest" ? "SHAP TreeExplainer attempted" : "Not available"}</strong></div>
          </div>
          <div className="metrics-block"><h3>Recorded held-out metrics</h3><div className="metric-chip-list">{Object.entries(model.metrics).map(([key, value]) => <span key={key}>{metricLabels[key] ?? key}<strong>{value === null ? "Not available" : value.toFixed(3)}</strong></span>)}</div><p>{model.metrics_source} This app does not reevaluate a held-out test set.</p></div>
          {model.graphs.length > 0 && <div className="graph-gallery"><div className="graph-gallery-title"><h3>Saved training/evaluation plots</h3><span>Original project artifacts</span></div><div className="graph-grid">{model.graphs.map((src) => { const assetUrl = `${apiBase}${src.replace(/^\/api/, "")}`; return <a key={src} href={assetUrl} target="_blank" rel="noreferrer" className="graph-image"><img src={assetUrl} alt={`${model.name} saved plot: ${src.split("/").at(-1)}`} loading="lazy" /><span>{src.split("/").at(-1)?.replaceAll("_", " ").replace(".png", "")} <ExternalLink size={13} /></span></a>; })}</div></div>}
        </article>
      ))}
    </>
  );
}
