import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { ModelsResponse, PatientFeatures, Prediction } from "../types";
import AssessmentPage from "./AssessmentPage";

vi.mock("../api/client", () => ({
  api: {
    models: vi.fn(),
    predict: vi.fn(),
  },
}));

const modelResponse: ModelsResponse = {
  default_model_id: "random_forest",
  models: [{
    id: "random_forest",
    name: "Random Forest",
    architecture: "Random Forest",
    description: "Fitted pipeline",
    available: true,
    error: null,
    is_default: true,
    threshold: 0.5,
    metrics: {},
    metrics_source: "Saved",
    graphs: [],
    input_features: [],
    explanation: "SHAP",
  }],
  disclaimer: "Research only",
};

const prediction: Prediction = {
  model_id: "random_forest",
  predicted_class: 0,
  predicted_label: "Model classified as disease absent",
  model_score: 0.2,
  threshold: 0.5,
  missing_features: [],
  explanation_status: "Unavailable",
  explanations: null,
  disclaimer: "Research only",
};

describe("assessment form submission", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.models).mockResolvedValue(modelResponse);
    vi.mocked(api.predict).mockResolvedValue(prediction);
  });

  it("sends unknown numeric and boolean answers as null", async () => {
    render(
      <MemoryRouter initialEntries={["/assessment"]}>
        <Routes>
          <Route path="/assessment" element={<AssessmentPage />} />
          <Route path="/result" element={<h1>Assessment result</h1>} />
        </Routes>
      </MemoryRouter>,
    );

    fireEvent.change(await screen.findByRole("spinbutton", { name: /Age/ }), {
      target: { value: "63" },
    });
    fireEvent.change(screen.getByRole("combobox", { name: /Sex recorded/ }), {
      target: { value: "Male" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Continue/ }));

    fireEvent.change(await screen.findByRole("combobox", { name: /Chest pain category/ }), {
      target: { value: "typical angina" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Continue/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Continue/ }));

    expect((await screen.findAllByText("Not available")).length).toBeGreaterThan(0);
    expect(api.predict).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Run assessment/ }));
    await screen.findByRole("heading", { name: "Assessment result" });

    const submitted = vi.mocked(api.predict).mock.calls[0][1] as PatientFeatures;
    expect(submitted.age).toBe(63);
    expect(submitted.sex).toBe("Male");
    expect(submitted.cp).toBe("typical angina");
    expect(submitted.trestbps).toBeNull();
    expect(submitted.chol).toBeNull();
    expect(submitted.fbs).toBeNull();
    expect(submitted.exang).toBeNull();
    expect(submitted.ca).toBeNull();
    await waitFor(() => expect(api.predict).toHaveBeenCalledWith("random_forest", submitted));
  });
});
