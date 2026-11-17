import { Activity, ArrowDown, ArrowRight, ArrowUpRight, BrainCircuit, ClipboardList, FileCheck2, HeartPulse, LockKeyhole, ScanLine, ShieldCheck, SlidersHorizontal } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useResearch } from '../state';
import { ErrorNotice, Loading, ModelCard, SafetyNotice } from '../components/ui';
import './Home.css';

const workflow = [
  { icon: ClipboardList, title: 'Enter clinical inputs', text: 'Provide demographics, reported symptoms, and available test measurements.' },
  { icon: SlidersHorizontal, title: 'Prepare the data', text: 'The fitted training pipeline handles missing values and feature encoding.' },
  { icon: BrainCircuit, title: 'Run a trained model', text: 'Choose an available binary classifier for a real local assessment.' },
  { icon: FileCheck2, title: 'Explore the result', text: 'Review the model score, its limitations, and supported explanations.' },
];

export default function Home() {
  const { health, models, loading, error, refresh } = useResearch();
  const available = models.models.filter((model) => model.available);
  return <div className="landing-page">
    <section className="landing-scene" aria-labelledby="landing-title" id="overview">
      <header className="landing-header">
        <Link to="/" className="landing-brand" aria-label="CorVeritas overview"><Activity size={24} strokeWidth={2} /><span>CorVeritas</span></Link>
        <nav className="landing-nav" aria-label="Main navigation">
          <a href="#journey">Overview</a>
          <Link to="/assessment">Assessment</Link>
          <Link to="/models">Models</Link>
          <Link to="/about">Methodology</Link>
        </nav>
        <Link to="/assessment" className="landing-header-cta">Start assessment <ArrowUpRight size={14} /></Link>
      </header>

      <div className="landing-facts"><span><i />13 CLINICAL FEATURES</span><span><i />THREE MODEL APPROACHES</span><span><i />LOCAL INFERENCE</span></div>
      <p className="landing-intro">A clearer view of cardiovascular AI. Explore real models, transparent inputs, and the patterns behind each prediction.</p>

      <div className="landing-sculpture" aria-hidden="true">
        <div className="landing-arch" />
        <div className="landing-card">
          <div className="landing-card-top"><Activity size={20} /><span>CV / 01</span></div>
          <div className="landing-heart"><HeartPulse size={116} strokeWidth={1.1} /></div>
          <div className="landing-card-bottom"><span>CorVeritas</span><small>EXPLAINABLE BY DESIGN</small></div>
        </div>
        <div className="landing-reflection" />
      </div>

      <div className="landing-headline">
        <p className="landing-kicker"><span />EXPLAINABLE CARDIOVASCULAR AI</p>
        <h1 id="landing-title">Research.<br />With heart.</h1>
        <a className="landing-explore" href="#journey">Explore the workspace <ArrowDown size={15} /></a>
      </div>

      <div className="landing-action">
        <div className="landing-action-icon" aria-hidden="true"><HeartPulse size={31} strokeWidth={1.3} /></div>
        <div className="landing-action-copy"><span className={`landing-readiness ${health?.ready ? 'ready' : ''}`}><i />{loading ? 'Checking backend' : health?.ready ? 'Backend connected' : 'Backend not ready'}</span><p>Your next research question.</p><Link to="/assessment">Start assessment <ArrowUpRight size={15} /></Link></div>
      </div>

      <div className="landing-scene-footer"><span><LockKeyhole size={12} /> Local processing. No account required.</span><Link to="/about"><ShieldCheck size={13} /><span>Research use only. Not clinically validated.</span><ArrowUpRight size={12} /></Link></div>
    </section>

    <div className="home-details" id="journey">
      <div className="home-details-intro"><div><p className="eyebrow">THE RESEARCH WORKSPACE</p><h2>Research with a little more heart.</h2></div><p>An explainable AI workspace for cardiovascular disease assessment. Model predictions do not establish or exclude a diagnosis.</p></div>
      <div className="stats-grid"><div className="stat-card"><span className="stat-icon"><HeartPulse size={21} /></span><div><span className="stat-value">13 <small>clinical features</small></span><p>One consistent input schema</p></div></div><div className="stat-card"><span className="stat-icon blue"><BrainCircuit size={21} /></span><div><span className="stat-value">{loading ? '—' : available.length}<small> / 3 models available</small></span><p>Checked by the inference backend</p></div></div><div className="stat-card"><span className="stat-icon violet"><ScanLine size={21} /></span><div><span className="stat-value">Binary <small>classification</small></span><p>Disease-associated pattern detection</p></div></div></div>
      <section className="section-block"><div className="section-heading"><div><p className="eyebrow">THE ASSESSMENT JOURNEY</p><h2>Four steps. One transparent workflow.</h2></div><Link className="text-link" to="/about">How it works <ArrowUpRight size={16} /></Link></div><div className="workflow-grid">{workflow.map(({ icon: Icon, title, text }, index) => <article className="workflow-card" key={title}><div className="workflow-top"><span className="workflow-icon"><Icon size={23} strokeWidth={1.6} /></span><span className="step-number">0{index + 1}</span></div><h3>{title}</h3><p>{text}</p></article>)}</div></section>
      <section className="section-block"><div className="section-heading"><div><p className="eyebrow">THE MODEL WORKBENCH</p><h2>Three approaches. A shared research question.</h2></div><Link className="text-link" to="/models">Compare models <ArrowRight size={16} /></Link></div>{loading ? <Loading /> : error ? <ErrorNotice message={error} retry={() => void refresh()} /> : <div className="model-grid">{models.models.map((model, index) => <ModelCard key={model.id} model={model} index={index} />)}</div>}</section>
      <SafetyNotice />
    </div>
  </div>;
}
