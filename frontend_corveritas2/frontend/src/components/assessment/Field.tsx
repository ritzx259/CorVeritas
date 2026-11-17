import { HelpCircle } from 'lucide-react';
import type { UseFormReturn } from 'react-hook-form';
import type { AssessmentDraft, FeatureName } from '../../types';
import { fieldLabels } from '../../validation/assessment';

interface FieldProps {
  name: FeatureName; form: UseFormReturn<AssessmentDraft>; label?: string;
  help: string; options?: { value: string; label: string }[];
  unit?: string; min?: number; max?: number; step?: string; required?: boolean;
}
export default function Field({ name, form, label, help, options, unit, min, max, step = 'any', required = false }: FieldProps) {
  const error = form.formState.errors[name]?.message;
  return <div className={`form-field ${error ? 'has-error' : ''}`}>
    <label htmlFor={name}>{label ?? fieldLabels[name]}{required ? <span className="required-mark" aria-label="required"> *</span> : <span className="optional-label">Optional</span>}<span className="field-help-icon" tabIndex={0} role="img" aria-label={help} title={help}><HelpCircle size={14} /></span></label>
    {options ? <select id={name} {...form.register(name)} aria-invalid={!!error} aria-describedby={`${name}-help${error ? ` ${name}-error` : ''}`}><option value="">{required ? 'Select an option' : 'Unknown / not available'}</option>{options.map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}</select> : <div className="input-with-unit"><input type="number" id={name} {...form.register(name)} min={min} max={max} step={step} placeholder={required ? 'Enter age' : 'Unknown / not available'} aria-invalid={!!error} aria-describedby={`${name}-help${error ? ` ${name}-error` : ''}`} />{unit && <span>{unit}</span>}</div>}
    <p className="field-hint" id={`${name}-help`}>{help}</p>
    {error && <p className="field-error" role="alert" id={`${name}-error`}>{error}</p>}
  </div>;
}
