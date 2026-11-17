import { zodResolver } from "@hookform/resolvers/zod";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  CircleHelp,
  RotateCcw,
  Stethoscope,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useForm, type FieldPath, type UseFormRegister, type UseFormSetValue } from "react-hook-form";
import { useLocation, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import Loading from "../components/Loading";
import SafetyNotice from "../components/SafetyNotice";
import type { ModelsResponse, PatientFeatures, Prediction } from "../types";
import {
  assessmentSchema,
  emptyAssessment,
  toPatientFeatures,
  type AssessmentValues,
  type ValidAssessmentValues,
} from "../validation/assessment";

const steps = [
  { title: "Demographics", description: "Basic profile" },
  { title: "Symptoms & history", description: "Reported information" },
  { title: "Clinical measurements", description: "Test results" },
  { title: "Review & predict", description: "Confirm assessment" },
];

const stepFields: FieldPath<AssessmentValues>[][] = [
  ["age", "sex"],
  ["cp", "exang", "fbs"],
  ["trestbps", "chol", "thalch", "restecg", "oldpeak", "slope", "ca", "thal"],
];

const labels: Record<keyof AssessmentValues, string> = {
  age: "Age",
  sex: "Sex recorded in the dataset",
  cp: "Chest pain category",
  trestbps: "Resting blood pressure",
  chol: "Serum cholesterol",
  fbs: "Fasting blood sugar result",
  restecg: "Resting ECG result",
  thalch: "Maximum heart rate achieved",
  exang: "Exercise-induced angina",
  oldpeak: "ST depression",
  slope: "ST segment slope",
  ca: "Major vessels visualized",
  thal: "Thal result",
};

function parseNumber(value: unknown) {
  if (value === "" || value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseBoolean(value: unknown) {
  if (value === "" || value === null || value === undefined) return null;
  return value === "true";
}

function FieldError({ message }: { message?: string }) {
  return message ? <span className="field-error" role="alert">{message}</span> : null;
}

function FieldLabel({ children, hint }: { children: string; hint?: string }) {
  return <span className="field-label">{children}{hint && <span className="tooltip-wrap"><CircleHelp size={14} /><span className="tooltip-text">{hint}</span></span>}</span>;
}

function UnknownOption() {
  return <option value="">Unknown / Not available</option>;
}

function NumberField({
  name,
  label,
  hint,
  unit,
  integer = false,
  optional = false,
  isUnknown = false,
  register,
  setValue,
  error,
}: {
  name: "age" | "trestbps" | "chol" | "thalch" | "oldpeak" | "ca";
  label: string;
  hint?: string;
  unit?: string;
  integer?: boolean;
  optional?: boolean;
  isUnknown?: boolean;
  register: UseFormRegister<AssessmentValues>;
  setValue: UseFormSetValue<AssessmentValues>;
  error?: string;
}) {
  return (
    <div className="form-field">
      <label className="field-label" htmlFor={`field-${name}`}>
        {label}
        {hint && <span className="tooltip-wrap"><CircleHelp size={14} /><span className="tooltip-text">{hint}</span></span>}
      </label>
      <span className="input-unit-wrap">
        <input id={`field-${name}`} className="form-control" type="number" step={integer ? 1 : "any"} disabled={optional && isUnknown} {...register(name, { setValueAs: parseNumber })} />
        {unit && <span className="unit-tag">{unit}</span>}
      </span>
      {optional && <label className="unknown-toggle"><input type="checkbox" checked={isUnknown} onChange={(event) => { if (event.target.checked) setValue(name, null, { shouldDirty: true }); }} /><span>Unknown / Not available</span></label>}
      <FieldError message={error} />
    </div>
  );
}

export default function AssessmentPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const editFeatures = (location.state as { features?: AssessmentValues } | null)?.features;
  const [step, setStep] = useState(0);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [selectedModel, setSelectedModel] = useState("");
  const [serviceError, setServiceError] = useState("");
  const [submitError, setSubmitError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const {
    register,
    watch,
    trigger,
    setValue,
    reset,
    handleSubmit,
    formState: { errors },
  } = useForm<AssessmentValues, unknown, ValidAssessmentValues>({
    resolver: zodResolver(assessmentSchema),
    defaultValues: editFeatures ?? emptyAssessment,
    mode: "onTouched",
  });

  const values = watch();
  const availableModels = useMemo(
    () => models?.models.filter((item) => item.available) ?? [],
    [models],
  );

  useEffect(() => {
    api.models().then((data) => {
      setModels(data);
      const first = data.models.find((item) => item.available);
      setSelectedModel(data.default_model_id ?? first?.id ?? "");
    }).catch((reason: Error) => setServiceError(reason.message));
  }, []);

  const stepError = (name: keyof AssessmentValues) =>
    errors[name]?.message as string | undefined;

  async function goNext() {
    const valid = await trigger(stepFields[step]);
    if (valid) setStep((current) => current + 1);
  }

  const onSubmit = handleSubmit(async (data) => {
    if (!selectedModel) {
      setSubmitError("No trained model is available. Check the model registry and backend status.");
      return;
    }
    setSubmitError("");
    setSubmitting(true);
    try {
      const patientFeatures: PatientFeatures = toPatientFeatures(data);
      const prediction: Prediction = await api.predict(selectedModel, patientFeatures);
      navigate("/result", { state: { prediction, features: patientFeatures } });
    } catch (reason) {
      setSubmitError(reason instanceof Error ? reason.message : "Assessment failed. Please try again.");
    } finally {
      setSubmitting(false);
    }
  });

  if (!models && !serviceError) return <Loading label="Connecting to the local model registry…" />;

  return (
    <>
      <div className="page-title-row">
        <div><span className="section-kicker">PRIVATE SESSION · NO PATIENT IDENTIFIERS</span><h1>Cardiovascular assessment</h1><p>Enter the clinical features used by the research models. Leave unavailable measurements blank.</p></div>
        <button type="button" className="button button-outline reset-button" onClick={() => { reset(emptyAssessment); setStep(0); setSubmitError(""); }}><RotateCcw size={16} /> Reset form</button>
      </div>
      <SafetyNotice compact />
      {serviceError && <div className="inline-alert"><CircleHelp size={18} /><div><strong>Assessment service unavailable</strong><p>{serviceError}</p></div></div>}
      {availableModels.length === 0 && !serviceError && <div className="inline-alert"><Stethoscope size={18} /><div><strong>No model is ready for prediction</strong><p>Place the matching model artifacts in the project models folder and restart the API. See the model registry for training commands.</p></div></div>}

      <div className="assessment-layout">
        <aside className="step-rail" aria-label="Assessment progress">
          {steps.map((item, index) => (
            <button type="button" key={item.title} onClick={() => index < step && setStep(index)} className={`step-item ${index === step ? "active" : ""} ${index < step ? "complete" : ""}`}>
              <span className="step-number">{index < step ? <Check size={15} /> : `0${index + 1}`}</span>
              <span><strong>{item.title}</strong><small>{item.description}</small></span>
            </button>
          ))}
          <div className="step-help"><Stethoscope size={17} /><p>For medical interpretation, consult a qualified healthcare professional. This form does not replace care.</p></div>
        </aside>

        <form className="assessment-card" onSubmit={onSubmit} noValidate>
          <div className="assessment-card-heading"><div><span className="section-kicker">STEP 0{step + 1} OF 04</span><h2>{steps[step].title}</h2><p>{steps[step].description}</p></div><span className="step-count">{step + 1}<span> / 4</span></span></div>
          <div className="progress-track"><span style={{ width: `${((step + 1) / steps.length) * 100}%` }} /></div>

          {step === 0 && (
            <div className="form-grid form-grid-two">
              <NumberField name="age" label="Age" unit="years" integer hint="Age in years, as recorded for the source dataset." register={register} setValue={setValue} error={stepError("age")} />
              <label className="form-field"><FieldLabel hint="The dataset provides Male and Female categories only.">Sex recorded in dataset</FieldLabel><select className="form-control" {...register("sex", { setValueAs: (value) => value || null })}><UnknownOption /><option value="Male">Male</option><option value="Female">Female</option></select><FieldError message={stepError("sex")} /></label>
              <div className="form-note full-span"><CircleHelp size={16} /><p>Only age and the dataset's sex category are requested here. No name, contact information, or patient identifier is collected.</p></div>
            </div>
          )}

          {step === 1 && (
            <div className="form-grid form-grid-two">
              <label className="form-field full-span"><FieldLabel hint="Select the source dataset category. It describes the reported chest pain pattern.">Chest pain category</FieldLabel><select className="form-control" {...register("cp", { setValueAs: (value) => value || null })}><UnknownOption /><option value="typical angina">Typical angina</option><option value="atypical angina">Atypical angina</option><option value="non-anginal">Non-anginal</option><option value="asymptomatic">Asymptomatic</option></select><FieldError message={stepError("cp")} /></label>
              <label className="form-field"><FieldLabel hint="Reported/measured during an exercise test. This is not a self-diagnosis question.">Exercise-induced angina</FieldLabel><select className="form-control" {...register("exang", { setValueAs: parseBoolean })}><UnknownOption /><option value="true">Yes</option><option value="false">No</option></select></label>
              <label className="form-field"><FieldLabel hint="Laboratory result: whether fasting blood sugar exceeds 120 mg/dL.">Fasting blood sugar above 120 mg/dL</FieldLabel><select className="form-control" {...register("fbs", { setValueAs: parseBoolean })}><UnknownOption /><option value="true">Yes</option><option value="false">No</option></select></label>
              <div className="form-note full-span"><CircleHelp size={16} /><p>Symptoms are self-reported; fasting glucose status should come from a blood test. Unknown values are submitted as missing, not guessed.</p></div>
            </div>
          )}

          {step === 2 && (
            <div className="form-grid form-grid-two">
              <NumberField name="trestbps" label="Resting blood pressure" unit="mmHg" optional isUnknown={values.trestbps === null} hint="Resting blood pressure measurement, typically obtained by a clinician." register={register} setValue={setValue} error={stepError("trestbps")} />
              <NumberField name="chol" label="Serum cholesterol" unit="mg/dL" optional isUnknown={values.chol === null} hint="Laboratory cholesterol result." register={register} setValue={setValue} error={stepError("chol")} />
              <NumberField name="thalch" label="Maximum heart rate achieved" unit="bpm" optional isUnknown={values.thalch === null} hint="Maximum recorded heart rate during exercise testing." register={register} setValue={setValue} error={stepError("thalch")} />
              <label className="form-field"><FieldLabel hint="Clinical interpretation of an electrocardiogram recorded at rest.">Resting ECG result</FieldLabel><select className="form-control" {...register("restecg", { setValueAs: (value) => value || null })}><UnknownOption /><option value="normal">Normal</option><option value="st-t abnormality">ST-T abnormality</option><option value="lv hypertrophy">LV hypertrophy</option></select><FieldError message={stepError("restecg")} /></label>
              <NumberField name="oldpeak" label="ST depression" unit="mm" optional isUnknown={values.oldpeak === null} hint="ST-segment depression induced by exercise relative to rest." register={register} setValue={setValue} error={stepError("oldpeak")} />
              <label className="form-field"><FieldLabel hint="The slope at the peak exercise ST segment.">ST segment slope</FieldLabel><select className="form-control" {...register("slope", { setValueAs: (value) => value || null })}><UnknownOption /><option value="upsloping">Upsloping</option><option value="flat">Flat</option><option value="downsloping">Downsloping</option></select><FieldError message={stepError("slope")} /></label>
              <NumberField name="ca" label="Major vessels visualized" unit="vessels (0–3)" integer optional isUnknown={values.ca === null} hint="Number of major vessels visualized by fluoroscopy." register={register} setValue={setValue} error={stepError("ca")} />
              <label className="form-field"><FieldLabel hint="Dataset term for the thalassemia test category. 'Reversible' is sent with the dataset spelling.">Thal result</FieldLabel><select className="form-control" {...register("thal", { setValueAs: (value) => value || null })}><UnknownOption /><option value="normal">Normal</option><option value="fixed defect">Fixed defect</option><option value="reversable defect">Reversible defect</option></select><FieldError message={stepError("thal")} /></label>
              <div className="form-note full-span"><CircleHelp size={16} /><p>These values require clinical measurements or tests. Leave a field as “Unknown / Not available” if the result is not known; fitted model preprocessing will handle missing values.</p></div>
            </div>
          )}

          {step === 3 && (
            <div className="review-section">
              <div className="review-callout"><Check size={17} /><span>Review the entries below before sending them to the selected local model.</span></div>
              {steps.slice(0, 3).map((section, index) => (
                <section className="review-group" key={section.title}>
                  <div className="review-group-title"><h3>{section.title}</h3><button type="button" className="text-link" onClick={() => setStep(index)}>Edit <ArrowRight size={14} /></button></div>
                  <dl>
                    {stepFields[index].map((field) => (
                      <div key={field} className="review-row"><dt>{labels[field]}</dt><dd>{formatValue(field, values[field])}</dd></div>
                    ))}
                  </dl>
                </section>
              ))}
              <div className="form-field model-select"><FieldLabel hint="This model is chosen explicitly. If it is unavailable, the request will not be redirected to another model.">Inference model</FieldLabel><select className="form-control" value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)} disabled={availableModels.length <= 1}>{availableModels.length === 0 && <option value="" disabled>No trained model available</option>}{availableModels.map((model) => <option key={model.id} value={model.id}>{model.name}{model.is_default ? " · default" : ""}</option>)}</select></div>
              {Object.values(values).some((value) => value === null) && <div className="missing-note"><CircleHelp size={16} /> Some values are unavailable. The fitted preprocessing pipeline will impute supported missing values; no arbitrary defaults are inserted.</div>}
              {submitError && <div className="error-banner" role="alert">{submitError}</div>}
            </div>
          )}

          <div className="form-actions">
            {step > 0 ? <button type="button" className="button button-outline" onClick={() => setStep((current) => current - 1)}><ArrowLeft size={16} /> Back</button> : <span />}
            {step < 3 ? <button type="button" className="button button-primary" onClick={goNext}>Continue <ArrowRight size={16} /></button> : <button type="submit" className="button button-primary" disabled={submitting || !availableModels.length}>{submitting ? <><span className="button-spinner" /> Assessing…</> : <>Run assessment <ArrowRight size={16} /></>}</button>}
          </div>
        </form>
      </div>
    </>
  );
}

function formatValue(name: keyof AssessmentValues, value: AssessmentValues[keyof AssessmentValues]) {
  if (value === null || value === undefined) return <span className="unknown-value">Not available</span>;
  if (name === "fbs" || name === "exang") return value ? "Yes" : "No";
  if (name === "thal") return value === "reversable defect" ? "Reversible defect" : String(value);
  return String(value);
}
