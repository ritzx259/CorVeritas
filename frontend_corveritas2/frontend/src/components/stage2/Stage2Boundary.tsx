import { Component, type ReactNode } from 'react';

export default class Stage2Boundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    if (this.state.failed) return <section className="card stage2-panel" aria-label="Stage 2 unavailable"><h2>Stage 2 — Disease Type Classification</h2><p role="alert">Stage 2 is unavailable. Your Stage 1 result is preserved.</p><button type="button" className="button secondary" onClick={() => this.setState({ failed: false })}>Retry Stage 2</button></section>;
    return this.props.children;
  }
}
