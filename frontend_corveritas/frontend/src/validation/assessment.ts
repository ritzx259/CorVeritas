import { z } from "zod";

const nullableNumber = (min: number, max: number) =>
  z.number().min(min).max(max).nullable();

export const assessmentSchema = z.object({
  age: z.number().int().min(0, "Enter a non-negative age.").max(120).nullable(),
  sex: z.enum(["Male", "Female"]).nullable(),
  cp: z.enum(["typical angina", "atypical angina", "non-anginal", "asymptomatic"]).nullable(),
  trestbps: nullableNumber(0, 400),
  chol: nullableNumber(0, 1500),
  fbs: z.boolean().nullable(),
  restecg: z.enum(["normal", "st-t abnormality", "lv hypertrophy"]).nullable(),
  thalch: nullableNumber(0, 350),
  exang: z.boolean().nullable(),
  oldpeak: nullableNumber(-20, 20),
  slope: z.enum(["upsloping", "flat", "downsloping"]).nullable(),
  ca: z.number().int().min(0).max(3).nullable(),
  thal: z.enum(["normal", "fixed defect", "reversable defect"]).nullable(),
}).superRefine((values, context) => {
  if (values.age === null) {
    context.addIssue({ code: "custom", path: ["age"], message: "Age is required." });
  }
  if (values.sex === null) {
    context.addIssue({ code: "custom", path: ["sex"], message: "Choose a sex category." });
  }
  if (values.cp === null) {
    context.addIssue({ code: "custom", path: ["cp"], message: "Choose a chest pain category." });
  }
});

export type AssessmentValues = z.input<typeof assessmentSchema>;
export type ValidAssessmentValues = z.output<typeof assessmentSchema>;

export const emptyAssessment: AssessmentValues = {
  age: null,
  sex: null,
  cp: null,
  trestbps: null,
  chol: null,
  fbs: null,
  restecg: null,
  thalch: null,
  exang: null,
  oldpeak: null,
  slope: null,
  ca: null,
  thal: null,
};

export function toPatientFeatures(values: ValidAssessmentValues) {
  if (values.age === null || values.sex === null || values.cp === null) {
    throw new Error("Age, sex, and chest pain category are required.");
  }
  return { ...values, age: values.age, sex: values.sex, cp: values.cp };
}
