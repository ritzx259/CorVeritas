import { describe, expect, it } from "vitest";
import { assessmentSchema, emptyAssessment } from "./assessment";

describe("assessment validation", () => {
  it("requires age, sex, and chest pain category", () => {
    const parsed = assessmentSchema.safeParse(emptyAssessment);
    expect(parsed.success).toBe(false);
    if (!parsed.success) {
      expect(parsed.error.issues.map((issue) => issue.path[0])).toEqual(["age", "sex", "cp"]);
    }
  });

  it("accepts the exact source spelling and nullable test results", () => {
    const result = assessmentSchema.safeParse({
      ...emptyAssessment,
      age: 63,
      sex: "Male",
      cp: "non-anginal",
      thal: "reversable defect",
      fbs: false,
      chol: null,
    });
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.thal).toBe("reversable defect");
      expect(result.data.fbs).toBe(false);
      expect(result.data.chol).toBeNull();
    }
  });

  it("does not accept alternative categorical spellings", () => {
    const result = assessmentSchema.safeParse({
      ...emptyAssessment,
      age: 63,
      sex: "Male",
      cp: "non angina",
    });
    expect(result.success).toBe(false);
  });

  it("enforces the vessel count as an integer from zero to three", () => {
    const values = { ...emptyAssessment, age: 63, sex: "Male" as const, cp: "asymptomatic" as const, ca: 4 };
    expect(assessmentSchema.safeParse(values).success).toBe(false);
  });
});
