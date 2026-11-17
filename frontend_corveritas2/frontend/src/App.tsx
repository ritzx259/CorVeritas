import { Activity, ArrowUpRight, BookOpen, ChartNoAxesCombined, ChevronRight, FlaskConical, HeartPulse, LayoutDashboard, Menu, PanelLeftClose, Plus, ShieldCheck, X } from 'lucide-react';
import { useState, useEffect, lazy, Suspense } from 'react';
import { NavLink, Link, Route, Routes, useLocation } from 'react-router-dom';
import { useResearch } from './state';
import { EmptyState, Loading } from './components/ui';

const Home = lazy(() => import('./pages/Home'));
const Assessment = lazy(() => import('./pages/Assessment'));
const Results = lazy(() => import('./pages/Results'));
const Models = lazy(() => import('./pages/Models'));
const About = lazy(() => import('./pages/About'));

const navigation = [
  { to: '/', title: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/assessment', title: 'Patient assessment', icon: HeartPulse },
  { to: '/models', title: 'Model comparison', icon: ChartNoAxesCombined },
  { to: '/about', title: 'Methodology', icon: BookOpen },
];
export default function App() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const { health, loading } = useResearch();
  const location = useLocation();
  const isHome = location.pathname === '/';
  useEffect(() => { setMobileOpen(false); window.scrollTo({ top: 0, behavior: 'instant' }); }, [location.pathname]);
  const currentTitle = location.pathname === '/results' ? 'Assessment results' : navigation.find((item) => item.to === location.pathname)?.title ?? 'Research workspace';
  return <div className={`app-shell ${isHome ? 'landing-shell' : ''}`}>
    <a className="skip-link" href="#main-content">Skip to main content</a>
    {!isHome && <>
    {mobileOpen && <button className="sidebar-overlay" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
    <aside className={`sidebar ${mobileOpen ? 'open' : ''}`}>
      <Link to="/" className="brand"><span className="brand-icon"><Activity size={27} strokeWidth={1.8} /></span><span>CorVeritas<small>CARDIOVASCULAR AI</small></span></Link>
      <button className="mobile-close icon-button" aria-label="Close navigation" onClick={() => setMobileOpen(false)}><X size={20} /></button>
      <div className="workspace-label">RESEARCH WORKSPACE <PanelLeftClose size={14} /></div>
      <nav aria-label="Main navigation">{navigation.map(({ to, title, icon: Icon, end }) => <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}><Icon size={19} strokeWidth={1.8} /><span>{title}</span>{to === '/assessment' && <Plus size={15} className="nav-plus" />}</NavLink>)}</nav>
      <div className="sidebar-bottom"><div className="research-note"><span className="research-note-icon"><FlaskConical size={21} /></span><h3>Built for discovery.</h3><p>An explainable approach to cardiovascular AI research.</p><Link to="/about">Explore the methodology <ArrowUpRight size={14} /></Link></div><div className="sidebar-footer"><ShieldCheck size={16} /><span>Educational & research use</span></div></div>
    </aside>
    </>}
    <div className="workspace">
      {!isHome && <>
      <header className="topbar"><div className="breadcrumb"><button className="icon-button mobile-menu" aria-label="Open navigation" onClick={() => setMobileOpen(true)}><Menu size={21} /></button><span>Workspace</span><ChevronRight size={14} /><strong>{currentTitle}</strong></div><div className="topbar-right"><span className={`connection ${health?.ready ? 'online' : ''}`}><span className="status-dot" />{loading ? 'Connecting' : health?.ready ? 'Backend connected' : 'Backend not ready'}</span><span className="research-tag">RESEARCH PROTOTYPE</span></div></header>
      <div className="research-safety-strip" role="note"><ShieldCheck size={16} /><span><strong>Research use only.</strong> Not clinically validated. Predictions do not establish or exclude a diagnosis.</span><Link to="/about">Learn more <ArrowUpRight size={13} /></Link></div>
      </>}
      <main id="main-content" tabIndex={-1}><Suspense fallback={<Loading label="Opening research workspace…" />}><Routes><Route path="/" element={<Home />} /><Route path="/assessment" element={<Assessment />} /><Route path="/results" element={<Results />} /><Route path="/models" element={<Models />} /><Route path="/about" element={<About />} /><Route path="*" element={<EmptyState title="Page not found"><Link to="/" className="button primary">Return to overview</Link></EmptyState>} /></Routes></Suspense></main>
      <footer className="workspace-footer"><span>CorVeritas <span className="footer-divider">/</span> Explainable cardiovascular AI</span><span>Local inference <span className="tiny-dot" /> No patient data stored by default</span></footer>
    </div>
  </div>;
}
