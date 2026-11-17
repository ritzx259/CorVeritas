import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import { ResearchProvider, AssessmentProvider } from './state';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter><ResearchProvider><AssessmentProvider><App /></AssessmentProvider></ResearchProvider></BrowserRouter>
  </React.StrictMode>,
);
