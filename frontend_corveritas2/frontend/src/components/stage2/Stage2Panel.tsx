import { BrainCircuit, CheckCircle2, Info, LoaderCircle, RefreshCw } from 'lucide-react';
import { useEffect, useRef, useState, type FormEvent } from 'react';
import type { AssessmentResult } from '../../types';
import { stage2Api } from '../../api/stage2';
import { ErrorNotice, Loading } from '../ui';
import ExplanationChart from '../charts/ExplanationChart';
import { reusedStage1Values, stage2Inputs, type Stage2Prediction, type Stage2Schema } from './types';
import './stage2.css';

type Phase = 'loading' | 'ready' | 'missing' | 'running' | 'completed' | 'unavailable' | 'failed';
const phaseNames: Record<Phase, string> = { loading: 'Checking availability', ready: 'Ready to classify', missing: 'Review missing or invalid inputs', running: 'Classification in progress', completed: 'Classification completed', unavailable: 'Classification unavailable', failed: 'Classification failed' };
const explanationLabels: Record<string, string> = { bmi: 'BMI', systolic_bp: 'Systolic BP', diastolic_bp: 'Diastolic BP', total_cholesterol: 'Total cholesterol', diabetes_history: 'Diabetes history', smoking_history: 'Lifetime smoking' };

export default function Stage2Panel({ result }: { result: AssessmentResult }) {
  const eligible = result.response.predicted_class === 1 && Number(result.features.age) >= 20;
  const [schema, setSchema] = useState<Stage2Schema | null>(null);
  const [phase, setPhase] = useState<Phase>('loading');
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Stage2Prediction | null>(null);
  const [attempt, setAttempt] = useState(0);
  const generation = useRef(0);
  useEffect(() => {
    const current = ++generation.current;
    setSchema(null); setPrediction(null); setDraft({}); setErrors({}); setError(null);
    if (!eligible) return;
    setPhase('loading');
    stage2Api.schema().then((metadata) => {
      if (current !== generation.current) return;
      setSchema(metadata);
      setPhase(metadata.available ? 'ready' : 'unavailable');
      if (!metadata.available) setError(metadata.reason ?? 'Stage 2 model files or its isolated environment are unavailable.');
    }).catch((reason) => {
      if (current !== generation.current) return;
      setPhase('unavailable'); setError(reason instanceof Error ? reason.message : 'Stage 2 could not be reached.');
    });
    return () => { generation.current++; };
  }, [result, eligible, attempt]);

  const fields = schema?.features ?? [];
  const reused = reusedStage1Values(result, fields);
  const additionalFields = fields.filter((field) => !(field.name in reused));
  const missingRequired = additionalFields.filter((field) => field.required && !(draft[field.name] ?? '').trim());
  const status = eligible ? missingRequired.length && phase === 'ready' ? 'Missing required inputs' : phaseNames[phase] : 'Not eligible for classification';
  const running = phase === 'running';
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!eligible || !schema?.available || running) return;
    const validated = stage2Inputs(fields, reused, draft);
    setErrors(validated.errors); setError(null);
    if (Object.keys(validated.errors).length) { setPhase('missing'); return; }
    const current = generation.current;
    setPhase('running'); setPrediction(null);
    try {
      const response = await stage2Api.predict(result, validated.additional);
      if (current !== generation.current) return;
      setPrediction(response); setPhase('completed');
    } catch (reason) {
      if (current !== generation.current) return;
      setPhase('failed'); setError(reason instanceof Error ? reason.message : 'Stage 2 failed. Your Stage 1 result is preserved.');
    }
  };
  const above = prediction?.conditions.filter((condition) => condition.binary_classification === 1) ?? [];
  return <section className="card stage2-panel" aria-labelledby="stage2-title" aria-busy={running}>
    <div className="section-heading"><div><p className="eyebrow">DISEASE TYPE CLASSIFICATION — STAGE 2</p><h2 id="stage2-title">Stage 2 — Disease Type Classification</h2></div><span className="stage2-state" role="status">{running ? <LoaderCircle className="spin" size={15} /> : phase === 'completed' && eligible ? <CheckCircle2 size={15} /> : <BrainCircuit size={15} />}{status}</span></div>
    <p className="stage2-subtitle">Identify the predicted cardiovascular disease category using the trained Stage 2 model.</p>
    {result.response.predicted_class !== 1 ? <div className="inline-note"><Info size={18} /><div><strong>Disease type classification was not performed because Stage 1 did not indicate cardiovascular disease.</strong><p>A negative Stage 1 result does not guarantee that disease is absent.</p></div></div> : Number(result.features.age) < 20 ? <div className="inline-note"><Info size={18} /><div><strong>Stage 2 is available only for adults aged 20–120.</strong><p>The Stage 2 training population starts at age 20. Your existing Stage 1 result is preserved.</p></div></div> : <>
      {phase === 'loading' && <Loading label="Checking the isolated Stage 2 model…" />}
      {schema?.available && <>
        <div className="stage2-model"><BrainCircuit size={19} /><div><strong>{schema.model_name}</strong><span>Selected in the saved model metadata · {schema.run_id}</span></div></div>
        <div className="inline-note"><Info size={18} /><p>This model estimates <strong>reported diagnosis histories</strong>: {schema.conditions?.map((condition) => condition.name).join('; ')}. The outputs are independent and may overlap; scores do not sum to 100%. They do not diagnose a current condition or identify one exclusive disease subtype.</p></div>
        {Object.keys(reused).length > 0 && <div className="stage2-reused"><h3>Reused from Stage 1</h3><dl>{fields.filter((field) => field.name in reused).map((field) => <div key={field.name}><dt>{field.label}</dt><dd>{reused[field.name]} {field.unit}</dd></div>)}</dl><p>Only compatible recorded values are reused. Stage 2 applies its own fitted preprocessing. A fasting-glucose threshold is not reused as a diabetes diagnosis.</p></div>}
        <form onSubmit={submit} noValidate>
          {additionalFields.length > 0 && <><h3>Additional Stage 2 inputs</h3><p className="stage2-input-help">Only values missing or outside Stage 2’s supported range are requested. Optional values can stay unknown; the saved Stage 2 preprocessing handles them.</p><div className="form-grid">{additionalFields.map((field) => <div key={field.name} className={`form-field ${errors[field.name] ? 'has-error' : ''}`}>
            <label htmlFor={`stage2-${field.name}`}>{field.label}{field.required ? <span className="required-mark">*</span> : <span className="optional-label">Optional</span>}</label>
            {field.type === 'category' ? <select id={`stage2-${field.name}`} value={draft[field.name] ?? ''} disabled={running} aria-invalid={Boolean(errors[field.name])} aria-describedby={`stage2-${field.name}-help${errors[field.name] ? ` stage2-${field.name}-error` : ''}`} onChange={(event) => { setDraft({ ...draft, [field.name]: event.target.value }); setPrediction(null); setPhase('ready'); setErrors({ ...errors, [field.name]: '' }); }}><option value="">Unknown / not provided</option>{field.categories?.map((category) => <option value={category} key={category}>{category}</option>)}</select> : <div className="input-with-unit"><input id={`stage2-${field.name}`} type="number" step="any" min={field.min} max={field.max} value={draft[field.name] ?? ''} placeholder="Unknown / not provided" disabled={running} aria-invalid={Boolean(errors[field.name])} aria-describedby={`stage2-${field.name}-help${errors[field.name] ? ` stage2-${field.name}-error` : ''}`} onChange={(event) => { setDraft({ ...draft, [field.name]: event.target.value }); setPrediction(null); setPhase('ready'); setErrors({ ...errors, [field.name]: '' }); }} />{field.unit && <span>{field.unit}</span>}</div>}
            <p className="field-hint" id={`stage2-${field.name}-help`}>{field.help}</p>{errors[field.name] && <p className="field-error" id={`stage2-${field.name}-error`}>{errors[field.name]}</p>}
          </div>)}</div></>}
          <div className="stage2-actions"><button type="submit" className="button primary" disabled={running || missingRequired.length > 0}>{running ? <><LoaderCircle size={17} className="spin" />Classifying…</> : <><BrainCircuit size={17} />Classify Disease Type</>}</button><p>Stage 1 stays visible and unchanged. Classification runs only when you choose this button.</p></div>
        </form>
      </>}
      {error && <ErrorNotice message={error} retry={phase === 'unavailable' ? () => setAttempt((value) => value + 1) : undefined} />}
      {phase === 'unavailable' && <button type="button" className="button secondary stage2-retry" onClick={() => setAttempt((value) => value + 1)}><RefreshCw size={15} />Retry Stage 2 availability</button>}
      {prediction && <div className="stage2-results" aria-live="polite">
        <h3>Stage 2 classification result</h3><p className="stage2-result-summary">{above.length ? `Above the saved research thresholds: ${above.map((condition) => condition.condition_name).join('; ')}.` : 'No supported category exceeded its saved research threshold.'} This does not override Stage 1 or establish or exclude a diagnosis.</p>
        <p className="stage2-score-note">Uncalibrated model scores, not clinical confidence probabilities. Each condition uses its own validation-selected threshold.</p>
        <div className="stage2-categories">{prediction.conditions.map((condition) => <article className="stage2-condition" key={condition.condition}>
          <div className="stage2-condition-heading"><h3>{condition.condition_name}</h3><span className={`stage2-class ${condition.binary_classification ? 'above' : ''}`}>{condition.binary_classification ? 'Above threshold' : 'Below threshold'}</span></div>
          <div className="stage2-score-value"><strong>{(condition.model_score * 100).toFixed(1)}%</strong><span>model score · threshold {(condition.selected_threshold * 100).toFixed(1)}%</span></div>
          <div className={`score-track stage2-score-track ${condition.binary_classification ? 'above' : ''}`} role="img" aria-label={`${condition.condition_name}: model score ${(condition.model_score * 100).toFixed(1)} percent, threshold ${(condition.selected_threshold * 100).toFixed(1)} percent`}><span style={{ width: `${condition.model_score * 100}%` }} /><i style={{ left: `${condition.selected_threshold * 100}%` }} /></div>
          {condition.feature_contributions?.length && prediction.explanation_method ? <details className="stage2-explanation"><summary>Explain this model score</summary><p>{prediction.explanation_method}</p><ExplanationChart contributions={condition.feature_contributions.map((entry) => ({ feature: entry.feature, label: explanationLabels[entry.feature] ?? fields.find((field) => field.name === entry.feature)?.label ?? entry.feature, value: entry.contribution }))} /><p>Baseline {condition.explanation_base_score?.toFixed(4)} + feature contributions = model score {condition.model_score.toFixed(4)}.</p>{condition.explanation_reconstruction_error !== undefined && <p>Reconstruction error: {condition.explanation_reconstruction_error.toExponential(2)}.</p>}<p>{prediction.explanation_limitations}</p></details> : <p className="stage2-score-note">Explanation currently unavailable.</p>}
        </article>)}</div>
        {prediction.warnings.length > 0 && <div className="notice warning"><Info size={18} /><div><strong>Stage 2 input and explanation notes</strong>{prediction.warnings.map((warning) => <p key={warning}>{warning}</p>)}</div></div>}
        <details className="stage2-limitations"><summary>Stage 2 model limitations and provenance</summary><p>{prediction.model_name} · {prediction.model_version} · Run {prediction.run_id}</p><ul>{prediction.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul></details>
      </div>}
    </>}
  </section>;
}
