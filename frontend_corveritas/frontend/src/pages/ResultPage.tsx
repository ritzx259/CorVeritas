import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  CircleHelp,
  ClipboardCheck,
  ShieldAlert,
  Sparkles,
} from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import SafetyNotice from "../components/SafetyNotice";
import type { PatientFeatures, Prediction } from "../types";

type ResultState = { prediction: Prediction; features: PatientFeatures };

const display: Record<keyof PatientFeatures, string> = {
  age: "Age",
  sex: "Sex",
  cp: "Chest pain category",
  trestbps: "Resting blood pressure",
  chol: "Serum cholesterol",
  fbs: "Fasting blood sugar >120 mg/dL",
  restecg: "Resting ECG",
  thalch: "Maximum heart rate",
  exang: "Exercise-induced angina",
  oldpeak: "ST depression",
  slope: "ST segment slope",
  ca: "Major vessels visualized",
  thal: "Thal result",
};

export default function ResultPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as ResultState | null;

  if (!state?.prediction || !state.features) {
    return <div className="empty-state"><CircleHelp size={28} /><h1>No assessment to display</h1><p>Run an assessment to see a result from a loaded model.</p><button className="button button-primary" onClick={() => navigate("/assessment")}>Start an assessment <ArrowRight size={16} /></button></div>;
  }

  const { prediction, features } = state;
  const modelName = prediction.model_id.replaceAll("_", " ");
  const chartData = [
    { name: "Model score", value: prediction.model_score },
    { name: "Remaining scale", value: 1 - prediction.model_score },
  ];

  return (
    <>
      <div className="page-title-row">
        <div><span className="section-kicker">LOCAL MODEL INFERENCE · COMPLETE</span><h1>Assessment result</h1><p>The following output was returned by the selected trained model.</p></div>
        <button className="button button-outline" onClick={() => navigate("/assessment", { state: { features } })}><ArrowLeft size={16} /> Edit assessment</button>
      </div>
      <SafetyNotice compact />
      <section className={`result-banner ${prediction.predicted_class === 1 ? "result-positive" : "result-negative"}`}>
        <div className="result-symbol">{prediction.predicted_class === 1 ? <ShieldAlert size={24} /> : <CheckCircle2 size={24} />}</div>
        <div className="result-main"><span className="section-kicker">BINARY MODEL CLASSIFICATION</span><h2>{prediction.predicted_class === 1 ? "Disease-present class" : "Disease-absent class"}</h2><p>{prediction.predicted_label}. This is a model classification, not a diagnosis.</p></div>
        <div className="result-model-tag"><span>Selected model</span><strong>{modelName}</strong></div>
      </section>

      <div className="result-grid">
        <section className="content-card score-card">
          <div className="card-heading"><div><span className="section-kicker">MODEL OUTPUT</span><h2>Score & decision threshold</h2></div><CircleHelp size={18} /></div>
          <div className="score-chart">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={chartData} dataKey="value" nameKey="name" innerRadius={69} outerRadius={88} startAngle={90} endAngle={-270} paddingAngle={0} stroke="none">
                  <Cell fill="#168c83" /><Cell fill="#e8eef1" />
                </Pie>
                <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
              </PieChart>
            </ResponsiveContainer>
            <div className="score-center"><strong>{(prediction.model_score * 100).toFixed(1)}%</strong><span>model score</span></div>
          </div>
          <div className="score-facts"><div><span>Decision threshold</span><strong>{prediction.threshold.toFixed(2)}</strong></div><div><span>Threshold result</span><strong>{prediction.predicted_class === 1 ? "At / above" : "Below"}</strong></div></div>
          <p className="score-caveat">This is the model's output score on a 0–1 scale. It is not a verified clinical risk probability and may not be calibrated.</p>
        </section>

        <section className="content-card">
          <div className="card-heading"><div><span className="section-kicker">MODEL TRANSPARENCY</span><h2>Patient-level explanation</h2></div><Sparkles size={18} /></div>
          <p className="explanation-status">{prediction.explanation_status}</p>
          {prediction.explanations?.length ? (
            <>
              <div className="contribution-list">{prediction.explanations.map((item) => (
                <div className="contribution-row" key={item.feature}><span>{item.feature}</span><div className="contribution-track"><span className={item.contribution >= 0 ? "contribution-up" : "contribution-down"} style={{ width: `${Math.min(Math.abs(item.contribution) * 100 * 3, 100)}%` }} /></div><strong className={item.contribution >= 0 ? "text-up" : "text-down"}>{item.contribution >= 0 ? "+" : ""}{item.contribution.toFixed(3)}</strong></div>
              ))}</div>
              <p className="tiny-note">Positive contributions move the model output toward class 1; negative contributions move it toward class 0. These are model associations, not causes.</p>
            </>
          ) : <div className="unavailable-explanation"><CircleHelp size={18} /><p>No trustworthy patient-specific explanation was produced for this model. No rule-based explanation is substituted.</p></div>}
        </section>

        <section className="content-card input-summary">
          <div className="card-heading"><div><span className="section-kicker">INPUT REVIEW</span><h2>Submitted feature summary</h2></div><ClipboardCheck size={18} /></div>
          <dl className="summary-grid">{(Object.keys(display) as (keyof PatientFeatures)[]).map((key) => (
            <div key={key}><dt>{display[key]}</dt><dd>{valueLabel(key, features[key])}</dd></div>
          ))}</dl>
          {prediction.missing_features.length > 0 && <div className="missing-note"><CircleHelp size={16} /><span>Missing values: {prediction.missing_features.map((key) => display[key as keyof PatientFeatures] ?? key).join(", ")}. The fitted model pipeline handled imputation.</span></div>}
        </section>
      </div>
      <div className="result-actions"><button className="button button-outline" onClick={() => navigate("/assessment", { state: { features } })}>Edit assessment</button><button className="button button-primary" onClick={() => navigate("/assessment", { state: null })}>New assessment <ArrowRight size={16} /></button></div>
    </>
  );
}

function valueLabel(name: keyof PatientFeatures, value: PatientFeatures[keyof PatientFeatures]) {
  if (value === null || value === undefined) return <span className="unknown-value">Not available</span>;
  if (name === "fbs" || name === "exang") return value ? "Yes" : "No";
  if (name === "thal") return value === "reversable defect" ? "Reversible defect" : String(value);
  return String(value);
}
