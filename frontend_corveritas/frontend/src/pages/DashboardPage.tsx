import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  BrainCircuit,
  CheckCircle2,
  CircleHelp,
  Database,
  HeartPulse,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import Loading from "../components/Loading";
import SafetyNotice from "../components/SafetyNotice";
import type { Health, ModelsResponse } from "../types";

export default function DashboardPage() {
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([api.models(), api.health()])
      .then(([modelInfo, status]) => {
        setModels(modelInfo);
        setHealth(status);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  const available = models?.models.filter((model) => model.available).length ?? 0;

  return (
    <>
      <section className="hero-panel">
        <div className="hero-copy">
          <div className="eyebrow"><span className="eyebrow-mark"><Activity size={14} /></span> EXPLAINABLE AI · RESEARCH PROTOTYPE</div>
          <h1>Clinical data.<br /><span>Thoughtful insight.</span></h1>
          <p>A transparent demonstration of binary cardiovascular disease classification using locally loaded machine-learning models and the UCI Heart Disease dataset.</p>
          <div className="hero-actions">
            <Link className="button button-primary" to="/assessment">Start an assessment <ArrowRight size={17} /></Link>
            <Link className="button button-quiet" to="/about">Explore methodology</Link>
          </div>
          <div className="hero-assurance"><ShieldCheck size={15} /> No patient identity collected · No inputs stored</div>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="pulse-orbit orbit-one" />
          <div className="pulse-orbit orbit-two" />
          <div className="heart-glass"><HeartPulse size={82} strokeWidth={1.15} /></div>
          <div className="pulse-chip"><span className="pulse-line" /> Model inference</div>
          <div className="hero-label"><span className="live-dot" /> Local models only</div>
        </div>
      </section>

      <SafetyNotice compact />

      <section className="section-block">
        <div className="section-heading">
          <div><span className="section-kicker">SYSTEM SNAPSHOT</span><h2>Research environment</h2></div>
          <Link className="text-link" to="/models">View model registry <ArrowUpRight size={15} /></Link>
        </div>
        {error ? (
          <div className="inline-alert"><CircleHelp size={18} /><div><strong>Backend connection unavailable</strong><p>{error} Start the FastAPI service to load model status and run assessments.</p></div></div>
        ) : !models ? <Loading /> : (
          <div className="stat-grid">
            <div className="stat-card"><span className="stat-icon teal"><BrainCircuit size={19} /></span><div className="stat-value">{available}<small> / {models.models.length}</small></div><div className="stat-label">Models loaded successfully</div><div className="stat-foot">{health?.ready ? "Inference service ready" : "No model ready"}</div></div>
            <div className="stat-card"><span className="stat-icon blue"><Database size={19} /></span><div className="stat-value">13</div><div className="stat-label">Training input features</div><div className="stat-foot">Original UCI feature names</div></div>
            <div className="stat-card"><span className="stat-icon violet"><Activity size={19} /></span><div className="stat-value">Binary</div><div className="stat-label">Classification task</div><div className="stat-foot">Disease absent / present</div></div>
            <div className="stat-card"><span className="stat-icon amber"><ShieldAlert size={19} /></span><div className="stat-value">Research</div><div className="stat-label">Intended use</div><div className="stat-foot">Not a clinical risk calculator</div></div>
          </div>
        )}
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div><span className="section-kicker">HOW IT WORKS</span><h2>From measurements to model output</h2></div>
          <span className="muted-caption">Every prediction uses an actual local model artifact</span>
        </div>
        <div className="workflow-grid">
          {[
            { icon: HeartPulse, n: "01", title: "Enter clinical features", text: "Provide the same 13 demographic, symptom, and test-result fields used during training. Missing test values can remain unknown." },
            { icon: Database, n: "02", title: "Preprocess as trained", text: "The saved fitted transformers handle imputation and encoding. The forest receives the original feature table through its complete pipeline." },
            { icon: Sparkles, n: "03", title: "Review a binary estimate", text: "The selected model returns its score and threshold decision. Random Forest may provide validated local SHAP contributions." },
          ].map(({ icon: Icon, n, title, text }) => (
            <article className="workflow-card" key={n}>
              <div className="workflow-top"><span>{n}</span><Icon size={20} /></div>
              <h3>{title}</h3><p>{text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="binary-callout">
        <div className="callout-icon"><CheckCircle2 size={22} /></div>
        <div><strong>One honest question, not a diagnosis</strong><p>The current MLP, CNN, and Random Forest models predict only the dataset's binary target: no disease (0) versus disease present (1). They do not identify disease subtypes.</p></div>
        <Link to="/about" aria-label="Read limitations"><ArrowRight size={18} /></Link>
      </section>
    </>
  );
}
