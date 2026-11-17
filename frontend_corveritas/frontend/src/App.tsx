import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import AppShell from "./components/AppShell";
import Loading from "./components/Loading";

const AboutPage = lazy(() => import("./pages/AboutPage"));
const AssessmentPage = lazy(() => import("./pages/AssessmentPage"));
const DashboardPage = lazy(() => import("./pages/DashboardPage"));
const ModelsPage = lazy(() => import("./pages/ModelsPage"));
const ResultPage = lazy(() => import("./pages/ResultPage"));

export default function App() {
  return (
    <AppShell>
      <Suspense fallback={<Loading label="Opening workspace…" />}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/assessment" element={<AssessmentPage />} />
          <Route path="/result" element={<ResultPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </Suspense>
    </AppShell>
  );
}
