import { AlertCircle, ArrowRight, CheckCircle2, LoaderCircle, ShieldCheck } from 'lucide-react';
import { Link } from 'react-router-dom';
import type { ReactNode } from 'react';
import type { Features, ModelInfo } from '../types';
import { featureOrder, fieldLabels, formatValue } from '../validation/assessment';

export function PageHeading({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: ReactNode }) {
  return <div className="page-heading"><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p className="page-description">{description}</p></div>{action}</div>;
}
export function StatusBadge({ available, loading = false }: { available: boolean; loading?: boolean }) {
  return <span className={`status-badge ${loading ? 'neutral' : available ? 'available' : 'unavailable'}`}><span className="status-dot" />{loading ? 'Checking' : available ? 'Available' : 'Unavailable'}</span>;
}
export function Loading({ label = 'Connecting to the research backend…' }: { label?: string }) { return <div role="status" className="loading-state"><LoaderCircle className="spin" size={20} /><span>{label}</span></div>; }
export function ErrorNotice({ message, retry }: { message: string; retry?: () => void }) {
  return <div role="alert" className="notice error"><AlertCircle size={19} /><div><strong>Unable to complete this request</strong><p>{message}</p>{retry && <button type="button" className="text-button" onClick={retry}>Try again <ArrowRight size={14} /></button>}</div></div>;
}
export function SafetyNotice({ compact = false }: { compact?: boolean }) {
  return <div className={`notice safety ${compact ? 'compact' : ''}`}><ShieldCheck size={20} /><div><strong>Research insight. Professional interpretation.</strong><p>This educational prototype is not a clinically validated medical device. Predictions cannot establish or exclude a diagnosis, and scores may not be calibrated probabilities.{!compact && ' A qualified healthcare professional must interpret clinical findings.'} Acute concerning symptoms require immediate medical attention, independent of any result.</p></div></div>;
}
export function ModelCard({ model, index }: { model: ModelInfo; index: number }) {
  return <article className="model-card"><div className="model-card-top"><span className="model-number">0{index + 1}</span><StatusBadge available={model.available} /></div><h3>{model.name}</h3><p>{model.description || model.algorithm || 'Binary cardiovascular pattern classification'}</p><div className="model-card-bottom"><span>{model.explanation_method ? 'Local explanation supported' : 'Explanation unavailable'}</span><Link to="/models" aria-label={`View ${model.name} details`}><ArrowRight size={17} /></Link></div></article>;
}
export function InputSummary({ features }: { features: Features }) {
  return <dl className="input-summary">{featureOrder.map((name) => <div key={name}><dt>{fieldLabels[name]}</dt><dd className={features[name] === null ? 'unknown-value' : ''}>{formatValue(name, features[name])}</dd></div>)}</dl>;
}
export function EmptyState({ title, children }: { title: string; children: ReactNode }) { return <div className="empty-state"><AlertCircle size={28} /><h2>{title}</h2>{children}</div>; }
export function CheckLine({ children }: { children: ReactNode }) { return <p className="check-line"><CheckCircle2 size={16} />{children}</p>; }
