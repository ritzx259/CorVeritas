import { describe, expect, it } from 'vitest';
import { assessmentSchema, emptyDraft, featureOrder, toFeatures } from '../validation/assessment';

const draft = { ...emptyDraft, age: '54', sex: 'Female', cp: 'atypical angina' };
describe('assessment validation and request conversion', () => {
  it('requires age, recorded sex, and exact chest pain category', () => {
    expect(assessmentSchema.safeParse(emptyDraft).success).toBe(false);
    expect(assessmentSchema.safeParse(draft).success).toBe(true);
    expect(assessmentSchema.safeParse({ ...draft, age: '' }).success).toBe(false);
    expect(assessmentSchema.safeParse({ ...draft, cp: 'chest pain' }).success).toBe(false);
  });
  it('rejects a corrected category spelling that the saved encoder does not expect', () => {
    expect(assessmentSchema.safeParse({ ...draft, thal: 'reversible defect' }).success).toBe(false);
    expect(toFeatures({ ...draft, thal: 'reversable defect' }).thal).toBe('reversable defect');
  });
  it('keeps missing values as null, and preserves zero and false', () => {
    const features = toFeatures({ ...draft, fbs: 'false', exang: 'true', ca: '0', oldpeak: '0' });
    expect(features.chol).toBeNull();
    expect(features.restecg).toBeNull();
    expect(features.fbs).toBe(false);
    expect(features.exang).toBe(true);
    expect(features.ca).toBe(0);
    expect(features.oldpeak).toBe(0);
    expect(Object.keys(features)).toEqual(featureOrder);
  });
  it('rejects out-of-bounds and fractional integer values', () => {
    for (const invalid of [{ age: '17' }, { age: '41.5' }, { ca: '4' }, { ca: '1.5' }, { thalch: '251' }, { oldpeak: '16' }, { trestbps: '39' }, { chol: 'Infinity' }]) expect(assessmentSchema.safeParse({ ...draft, ...invalid }).success).toBe(false);
  });
});
