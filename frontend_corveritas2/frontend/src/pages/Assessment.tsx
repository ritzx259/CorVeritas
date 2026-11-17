import { ArrowLeft, ArrowRight, Check, ChevronRight, ClipboardCheck, FlaskConical, HeartPulse, Info, LoaderCircle, LockKeyhole, RotateCcw, ShieldCheck, Stethoscope, UserRound } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useForm, type Resolver } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import Field from '../components/assessment/Field';
import { ErrorNotice, InputSummary, PageHeading, SafetyNotice, StatusBadge } from '../components/ui';
import { useAssessment, useResearch } from '../state';
import { assessmentSchema, emptyDraft, fieldLabels, toFeatures } from '../validation/assessment';
import type { AssessmentDraft, FeatureName, Features } from '../types';

const steps = [
  { title: 'Demographics', subtitle: 'A little clinical context', icon: UserRound, fields: ['age', 'sex'] },
  { title: 'Symptoms & history', subtitle: 'Reported symptoms and test history', icon: HeartPulse, fields: ['cp', 'exang', 'fbs'] },
  { title: 'Clinical measurements', subtitle: 'Available clinical test results', icon: Stethoscope, fields: ['trestbps', 'chol', 'thalch', 'restecg', 'oldpeak', 'slope', 'ca', 'thal'] },
  { title: 'Review & assess', subtitle: 'Confirm inputs and choose a model', icon: ClipboardCheck, fields: [] },
];
const booleans = [{ value: 'true', label: 'Yes' }, { value: 'false', label: 'No' }];
const options = (values: string[]) => values.map((value) => ({ value, label: value.charAt(0).toUpperCase() + value.slice(1) }));

export default function Assessment() {
  const { draft, setDraft, setResult, selectedModel, setSelectedModel, reset: resetAssessment } = useAssessment();
  const { models, error: backendError, loading, refresh } = useResearch();
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const form = useForm<AssessmentDraft>({ defaultValues: draft, resolver: zodResolver(assessmentSchema) as Resolver<AssessmentDraft>, mode: 'onTouched' });
  const values = form.watch();
  const available = models.models.filter((model) => model.available);
  const model = available.find((item) => item.id === selectedModel);
  useEffect(() => {
    if (!selectedModel && models.default_model_id && available.some((item) => item.id === models.default_model_id)) setSelectedModel(models.default_model_id);
  }, [models, selectedModel, setSelectedModel, available]);
  useEffect(() => {
    const subscription = form.watch((next) => setDraft(next as AssessmentDraft));
    return () => subscription.unsubscribe();
  }, [form, setDraft]);
  const unknown = (Object.keys(values) as FeatureName[]).filter((name) => !['age', 'sex', 'cp'].includes(name) && values[name] === '');
  const moveNext = async () => {
    const valid = await form.trigger(steps[step].fields as FeatureName[], { shouldFocus: true });
    if (valid) { setStep((current) => Math.min(current + 1, 3)); setError(null); window.scrollTo({ top: 0, behavior: 'smooth' }); }
  };
  const submit = form.handleSubmit(async (data) => {
    if (!model) { setError('Select a model that has loaded successfully. Refresh backend status if no models are available.'); return; }
    setError(null); setSubmitting(true);
    try {
      const features = toFeatures(data);
      const response = await api.predict(model.id, features);
      setResult({ response, features, modelName: model.name });
      navigate('/results');
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'The assessment could not be completed. Please retry.'); }
    finally { setSubmitting(false); }
  }, () => { setError('Some inputs need attention. Return to the relevant section and review the highlighted fields.'); });
  const reset = () => { form.reset({ ...emptyDraft }); resetAssessment(); setStep(0); setError(null); };
  const reviewFeatures = Object.fromEntries(Object.entries(values).map(([key, value]) => [key, value === '' ? null : ['fbs', 'exang'].includes(key) ? value === 'true' : ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca'].includes(key) ? Number(value) : value])) as Features;
  return <>
    <PageHeading eyebrow="PATIENT ASSESSMENT" title="Begin with the complete picture." description="Enter the available information. Let the trained pipeline handle what is missing." action={<button type="button" className="button secondary small" onClick={reset} disabled={submitting}><RotateCcw size={15} />Reset form</button>} />
    <div className="assessment-layout">
      <aside className="assessment-rail"><div className="assessment-steps" aria-label="Assessment progress">{steps.map(({ title, subtitle, icon: Icon }, index) => <button key={title} type="button" className={`assessment-step ${step === index ? 'current' : ''} ${step > index ? 'complete' : ''}`} aria-current={step === index ? 'step' : undefined} disabled={index > step || submitting} onClick={() => setStep(index)}><span className="step-circle">{step > index ? <Check size={18} /> : <Icon size={18} />}</span><span><strong>{title}</strong><small>{subtitle}</small></span>{step === index && <ChevronRight size={15} />}</button>)}</div><div className="privacy-card"><LockKeyhole size={21} /><h3>Your inputs stay in this session.</h3><p>No name or contact details are needed. This app does not save assessment inputs by default. Refreshing the page clears the form.</p></div></aside>
      <section className="assessment-panel"><div className="form-progress"><span>STEP 0{step + 1} <span>OF 04</span></span><div><span style={{ width: `${(step + 1) * 25}%` }} /></div><span>{(step + 1) * 25}%</span></div><div className="form-heading"><span className="form-section-icon">{step === 0 ? <UserRound size={24} /> : step === 1 ? <HeartPulse size={24} /> : step === 2 ? <Stethoscope size={24} /> : <ClipboardCheck size={24} />}</span><div><h2>{steps[step].title}</h2><p>{step === 0 ? 'Start with the basic information used by the training dataset.' : step === 1 ? 'Distinguish reported symptoms from measurements requiring a clinical test.' : step === 2 ? 'Use documented clinical results. Leave unavailable measurements blank.' : 'Check your information before running a real model prediction.'}</p></div></div>
      <form onSubmit={(event) => { if (step < 3) { event.preventDefault(); void moveNext(); } else void submit(event); }} noValidate>
        {step === 0 && <><div className="inline-note"><Info size={17} /><span>Age, recorded sex, and chest pain category are required. All other fields can be unknown.</span></div><div className="form-grid"><Field name="age" form={form} help="Adult age in completed years (18–120). This prototype is not intended for children." min={18} max={120} step="1" unit="years" required /><Field name="sex" form={form} help="The dataset records sex as Male or Female. This limited representation is a model constraint." options={options(['Male', 'Female'])} required /></div><div className="dataset-note"><ShieldCheck size={18} /><p><strong>A note on the dataset</strong>The UCI source includes a limited historical population. Its categories do not represent every individual or clinical setting.</p></div></>}
        {step === 1 && <><div className="form-group-label">REPORTED SYMPTOMS</div><div className="form-grid"><Field name="cp" form={form} help="Select the documented chest pain category. Angina is chest discomfort associated with reduced blood supply to the heart." options={[{ value: 'typical angina', label: 'Typical angina' }, { value: 'atypical angina', label: 'Atypical angina' }, { value: 'non-anginal', label: 'Non-anginal pain' }, { value: 'asymptomatic', label: 'Asymptomatic' }]} required /><Field name="exang" form={form} help="Whether exercise induced angina during assessment. Use Unknown when this has not been established." options={booleans} /></div><div className="form-group-label with-border">LABORATORY TEST HISTORY</div><div className="form-grid"><Field name="fbs" form={form} label="Fasting blood sugar above threshold" help="Requires a fasting blood test: was glucose greater than 120 mg/dL? This is not a symptom or a diabetes diagnosis." options={[{ value: 'true', label: 'Yes — above 120 mg/dL' }, { value: 'false', label: 'No — 120 mg/dL or below' }]} /></div></>}
        {step === 2 && <><div className="inline-note"><FlaskConical size={18} /><span>These inputs require clinical measurement or medical testing. Do not guess. Blank numerical fields are sent as missing values.</span></div><div className="form-grid"><Field name="trestbps" form={form} help="Resting systolic blood pressure recorded at admission. Valid input: 40–300 mmHg." unit="mmHg" min={40} max={300} /><Field name="chol" form={form} help="Serum cholesterol from a laboratory result. Zero values exist in the source data and warrant clinical interpretation." unit="mg/dL" min={0} max={1000} /><Field name="thalch" form={form} help="The maximum heart rate achieved during the clinical assessment, not a guessed or age-based estimate." unit="bpm" min={30} max={250} /><Field name="restecg" form={form} help="The interpretation recorded from a resting electrocardiogram (ECG)." options={[{ value: 'normal', label: 'Normal' }, { value: 'st-t abnormality', label: 'ST–T wave abnormality' }, { value: 'lv hypertrophy', label: 'Left ventricular hypertrophy' }]} /><Field name="oldpeak" form={form} help="ST depression induced by exercise relative to rest, using the value reported in the source test (−5 to 15 mm)." unit="mm" min={-5} max={15} /><Field name="slope" form={form} help="The slope of the peak exercise ST segment, as interpreted on the clinical test." options={options(['upsloping', 'flat', 'downsloping'])} /><Field name="ca" form={form} help="Number of major vessels (0–3) colored by fluoroscopy. Requires a clinical imaging result." options={options(['0', '1', '2', '3'])} /><Field name="thal" form={form} help="The source dataset's thal perfusion-test category. This field is not a diagnosis of thalassemia." options={[{ value: 'normal', label: 'Normal' }, { value: 'fixed defect', label: 'Fixed defect' }, { value: 'reversable defect', label: 'Reversible defect' }]} /></div></>}
        {step === 3 && <><div className="review-topline"><h3>Assessment inputs</h3><button type="button" className="text-button" onClick={() => setStep(0)}>Edit inputs <ArrowRight size={14} /></button></div><InputSummary features={reviewFeatures} />{unknown.length > 0 && <div className="notice warning"><Info size={19} /><div><strong>{unknown.length} of 13 inputs are unavailable</strong><p>{unknown.map((name) => fieldLabels[name]).join(', ')}. The fitted model preprocessing will impute supported missing values. This may affect the result and its explanation.</p></div></div>}<div className="model-selection"><label htmlFor="model-choice">Assessment model</label><p>Only models loaded successfully by the backend can be selected.</p>{backendError && <ErrorNotice message={backendError} retry={() => void refresh()} />}<select id="model-choice" value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)} disabled={submitting || loading}><option value="">Select an available model</option>{available.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select>{model && <div className="chosen-model"><StatusBadge available /><span>Decision threshold: {model.threshold.toFixed(2)}</span></div>}{!loading && available.length === 0 && <p className="field-error">No trained model is ready. Open Model comparison for artifact status and training commands.</p>}</div><div className="inline-note"><Info size={18} /><span>The output is a binary model estimate. It does not identify a specific disease or replace clinical assessment.</span></div></>}
        {error && <ErrorNotice message={error} />}
        <div className="form-actions"><button type="button" className="button secondary" disabled={step === 0 || submitting} onClick={() => { setStep((current) => current - 1); setError(null); }}><ArrowLeft size={16} />Back</button><span className="required-note">{step < 3 ? '* Required fields' : 'Processed locally. Never a diagnosis.'}</span>{step < 3 ? <button type="submit" className="button primary">Continue <ArrowRight size={16} /></button> : <button type="submit" className="button primary" disabled={!model || submitting}>{submitting ? <><LoaderCircle size={17} className="spin" />Running model…</> : <><HeartPulse size={17} />Run assessment</>}</button>}</div>
      </form></section>
    </div><SafetyNotice compact />
  </>;
}
