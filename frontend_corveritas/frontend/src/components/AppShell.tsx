import {
  Activity,
  BookOpenText,
  ChevronRight,
  CircleHelp,
  HeartPulse,
  LayoutDashboard,
  Menu,
  ShieldCheck,
  X,
} from "lucide-react";
import { useState } from "react";
import { Link, NavLink } from "react-router-dom";
import type { ReactNode } from "react";

const links = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/assessment", label: "New assessment", icon: HeartPulse },
  { to: "/models", label: "Model registry", icon: Activity },
  { to: "/about", label: "Methodology", icon: BookOpenText },
];

export default function AppShell({ children }: { children: ReactNode }) {
  const [menuOpen, setMenuOpen] = useState(false);
  return (
    <div className="app-shell">
      <aside className={`sidebar ${menuOpen ? "sidebar-open" : ""}`}>
        <Link to="/" className="brand" onClick={() => setMenuOpen(false)}>
          <span className="brand-mark"><HeartPulse size={23} strokeWidth={2.3} /></span>
          <span><strong>CorVeritas</strong><small>AI cardiovascular research</small></span>
        </Link>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          {links.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end={to === "/"} onClick={() => setMenuOpen(false)}>
              <Icon size={18} />
              <span>{label}</span>
              {to === "/assessment" && <ChevronRight className="nav-chevron" size={15} />}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="privacy-card">
            <ShieldCheck size={17} />
            <span><strong>Private by design</strong><small>Inputs are not saved by this app.</small></span>
          </div>
          <Link className="sidebar-help" to="/about"><CircleHelp size={16} /> About this research prototype</Link>
        </div>
      </aside>
      {menuOpen && <button className="mobile-scrim" aria-label="Close menu" onClick={() => setMenuOpen(false)} />}
      <div className="main-column">
        <header className="topbar">
          <button className="mobile-menu icon-button" aria-label="Open menu" onClick={() => setMenuOpen((open) => !open)}>
            {menuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
          <div className="breadcrumb">Research workspace <span>/</span> <strong>Cardiovascular assessment</strong></div>
          <div className="topbar-right"><span className="live-dot" /> Local demo</div>
        </header>
        <main className="page-content">{children}</main>
        <footer className="footer-disclaimer">
          <ShieldCheck size={15} />
          <span>Educational research prototype · Not a medical device or a substitute for professional care</span>
        </footer>
      </div>
    </div>
  );
}
