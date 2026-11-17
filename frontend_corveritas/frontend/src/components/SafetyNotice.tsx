import { AlertTriangle } from "lucide-react";

export default function SafetyNotice({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`safety-notice ${compact ? "safety-compact" : ""}`} role="note">
      <AlertTriangle size={18} />
      <p>
        <strong>Research use only.</strong> This prototype is not clinically validated and cannot diagnose or rule out heart disease. Its score is not necessarily a calibrated probability. A qualified clinician must interpret results. Seek immediate medical attention for acute or concerning symptoms, regardless of any model output.
      </p>
    </div>
  );
}
