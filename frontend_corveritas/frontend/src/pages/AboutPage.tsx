import { ArrowRight, BookOpenText, Database, FileText, GitBranch, ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import SafetyNotice from "../components/SafetyNotice";

const methods = [
  { title: "Dataset and target", icon: Database, text: "The supplied heart_disease_uci.csv contains the UCI Heart Disease records. Training scripts derive a binary target from num: 0 maps to no disease; values greater than 0 map to disease present. Identifier, source-dataset, and original target columns are excluded from the 13 input features." },
  { title: "Preprocessing", icon: GitBranch, text: "The saved MLP and CNN preprocessors impute numeric values with the median, standardize numeric features, impute categorical values with the most frequent category, and one-hot encode categories. The Random Forest artifact contains its complete fitted preprocessing pipeline; it receives the raw ordered DataFrame directly." },
  { title: "Model training and evaluation", icon: FileText, text: "The MLP and CNN use an 80/20 stratified train/test split with random_state=42 and a validation split from training data. The Random Forest script also holds out 20% for testing, then uses stratified five-fold cross-validation and RandomizedSearchCV on the training portion. No test set is re-evaluated by this app." },
  { title: "Current model scope", icon: BookOpenText, text: "The MLP, 1D CNN and Random Forest are independent binary classifiers. Names such as stage2_cnn and stage3_random_forest identify project artifacts only; they do not classify stages or subtypes of disease. No subtype prediction is implemented." },
];

export default function AboutPage() {
  return (
    <>
      <div className="page-title-row"><div><span className="section-kicker">PROJECT NOTES</span><h1>About & methodology</h1><p>What CorVeritas currently does, how it processes data, and where its limits are.</p></div><span className="method-tag"><ShieldCheck size={15} /> Transparent by design</span></div>
      <SafetyNotice />
      <div className="method-grid">{methods.map(({ title, icon: Icon, text }, index) => <article className="method-card" key={title}><div className="method-card-head"><span className="method-index">0{index + 1}</span><span className="method-icon"><Icon size={19} /></span></div><h2>{title}</h2><p>{text}</p></article>)}</div>
      <section className="content-card xai-section">
        <span className="section-kicker">EXPLAINABLE AI</span><h2>Local contributions, not causal claims</h2>
        <p>When the Random Forest and its matching pipeline load, the API attempts a patient-specific SHAP TreeExplainer calculation. Encoded contributions are grouped back to original clinical feature names, and additivity is checked against the model output before any values are shown. SHAP values describe how the model output is attributed for that input under its tree-path-dependent assumptions; they do not describe medical causes. Neural-network predictions currently do not have a validated local explanation implementation and are labelled unavailable.</p>
        <p>Saved feature-importance charts are not treated as patient-specific explanations. If an explanation package is unavailable or the calculation fails validation, the result page says so rather than fabricating one.</p>
      </section>
      <section className="future-card">
        <div className="future-icon"><GitBranch size={20} /></div><div><span className="section-kicker">FUTURE RESEARCH DIRECTION</span><h2>A separate subtype model would be a new stage</h2><p>A future architecture could use a first binary classifier to estimate whether a record belongs to the disease-present class, then route eligible records to a separately trained and clinically evaluated subtype classifier. That second model and routing workflow are not part of this prototype.</p></div>
      </section>
      <div className="method-cta"><div><strong>Explore the loaded artifacts</strong><p>Review real availability and the saved plots.</p></div><Link className="button button-primary" to="/models">Open model registry <ArrowRight size={16} /></Link></div>
    </>
  );
}
