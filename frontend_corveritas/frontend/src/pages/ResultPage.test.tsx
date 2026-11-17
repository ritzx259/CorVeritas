import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import ResultPage from "./ResultPage";
import type { PatientFeatures, Prediction } from "../types";

const prediction: Prediction = {
  model_id: "random_forest",
  predicted_class: 0,
  predicted_label: "Model classified as disease absent",
  model_score: 0.24,
  threshold: 0.5,
  missing_features: ["chol"],
  explanation_status: "Explanation currently unavailable for this model.",
  explanations: null,
  disclaimer: "Research use only.",
};

const features: PatientFeatures = {
  age: 60,
  sex: "Female",
  cp: "atypical angina",
  trestbps: 120,
  chol: null,
  fbs: false,
  restecg: "normal",
  thalch: 145,
  exang: false,
  oldpeak: 0,
  slope: "flat",
  ca: 0,
  thal: "reversable defect",
};

describe("prediction result", () => {
  it("renders the model output, unavailable feature, and non-diagnostic caution", () => {
    render(
      <MemoryRouter initialEntries={[{ pathname: "/result", state: { prediction, features } }]}>
        <Routes><Route path="/result" element={<ResultPage />} /></Routes>
      </MemoryRouter>,
    );
    expect(screen.getByRole("heading", { name: "Disease-absent class" })).toBeInTheDocument();
    expect(screen.getByText("24.0%")).toBeInTheDocument();
    expect(screen.getByText(/not a diagnosis/i)).toBeInTheDocument();
    expect(screen.getByText("Serum cholesterol")).toBeInTheDocument();
    expect(screen.getByText("Explanation currently unavailable for this model.")).toBeInTheDocument();
  });
});
