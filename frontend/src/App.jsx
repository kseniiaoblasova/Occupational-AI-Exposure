import { useState, useEffect } from 'react'
import ModelSection from './components/ModelSection.jsx'

// TODO: replace with actual URLs once available
const GITHUB_URL = '#'
const MEDIUM_URL = '#'

export default function App() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <>
      {/* Paper grain overlay */}
      <div className="grain" aria-hidden="true" />

      {/* ── Navigation ──────────────────────────── */}
      <nav className={`nav${scrolled ? ' nav--elevated' : ''}`} role="navigation">
        <div className="nav-left">
          <a href={GITHUB_URL} className="nav-ext" aria-label="View project on GitHub">
            Github
          </a>
          <span className="nav-sep" aria-hidden="true">/</span>
          <a href={MEDIUM_URL} className="nav-ext" aria-label="Read article on Medium">
            Medium
          </a>
        </div>
        <div className="nav-right">
          <a href="#about" className="nav-link">About</a>
          <a href="#model" className="nav-link">Try model</a>
          <a href="#paper" className="nav-link">Paper</a>
          <a href="#tech" className="nav-link">Technical details</a>
          <a href="#sources" className="nav-link">Sources</a>
        </div>
      </nav>

      {/* ── Hero ────────────────────────────────── */}
      <section className="hero" aria-label="Introduction">
        <div className="hero-body">
          <div className="hero-rule" aria-hidden="true" />

          <h1 className="hero-title">
            <span className="hero-title-main">AI and Job Market:</span>
            <span className="hero-subtitle">
              <span>A Machine Learning Approach</span>
              <span>to Predicting Occupational Exposure</span>
            </span>
          </h1>

          <p className="hero-desc">
            Enter a job title. Get a displacement risk score — powered by O*NET task data,
            BLS wages, and a custom logistic regression trained on 756 U.S. occupations.
          </p>

          <div className="hero-actions">
            <a href="#model" className="btn-primary">
              Try the model
              <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
                <path d="M2.5 7.5h10M9 3.5l4 4-4 4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </a>
            <a href="#about" className="btn-ghost">Learn more</a>
          </div>
        </div>
      </section>

      {/* ── About ───────────────────────────────── */}
      <section id="about" className="section">
        <div className="section-inner">
          <div className="section-eyebrow">About</div>
          <div className="section-body">
            <h2 className="section-heading">The methodology</h2>
            <p className="section-text">
              This tool estimates how exposed a given occupation is to displacement by AI
              systems. It maps job titles to O*NET standard occupational codes via Claude,
              engineers features from task statements using keyword analysis, and runs a
              custom logistic regression trained on 756 occupations.
            </p>
            <p className="section-text">
              The model uses 11 features: O*NET bright-outlook and green-economy flags,
              Job Zone (education and experience requirements), BLS median annual wage,
              and seven task-category proportions — computer use, physical work,
              communication, analysis, management, creative work, and text-native tasks.
            </p>

            <div className="stat-row">
              <div className="stat">
                <span className="stat-n">756</span>
                <span className="stat-lbl">Occupations</span>
              </div>
              <div className="stat">
                <span className="stat-n">11</span>
                <span className="stat-lbl">Features</span>
              </div>
              <div className="stat">
                <span className="stat-n">3</span>
                <span className="stat-lbl">Data sources</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Try model ───────────────────────────── */}
      <section id="model" className="section section--tinted section--model">
        <ModelSection />
      </section>

      {/* ── Paper ───────────────────────────────── */}
      <section id="paper" className="section">
        <div className="section-inner">
          <div className="section-eyebrow">Paper</div>
          <div className="section-body">
            <h2 className="section-heading">The paper</h2>
            <p className="section-text section-text--dim">
              Full write-up and findings — coming soon.
            </p>
          </div>
        </div>
      </section>

      {/* ── Technical details ───────────────────── */}
      <section id="tech" className="section section--tinted">
        <div className="section-inner">
          <div className="section-eyebrow">Technical details</div>
          <div className="section-body">
            <h2 className="section-heading">How it's built</h2>
            <p className="section-text">
              The backend is a FastAPI service that loads a trained logistic regression model
              and a fitted scaler at startup. For each job title query it calls the
              Anthropic Claude API to resolve the title to an O*NET SOC code, fetches
              occupation features from the O*NET Web Services API, and retrieves median
              wages from the BLS OEWS public data API.
            </p>
            <p className="section-text">
              The logistic regression is implemented from scratch using gradient ascent
              on the log-likelihood with learning rate decay — not scikit-learn. The
              frontend is React + Vite. Feature engineering is regex-based: seven keyword
              patterns applied to O*NET task statements produce the <code>pct_*</code> proportions
              used as model inputs.
            </p>
            <p className="section-text section-text--dim">
              APIs used: Anthropic Claude, O*NET Web Services, BLS OEWS public data API.
            </p>
          </div>
        </div>
      </section>

      {/* ── Sources ─────────────────────────────── */}
      <section id="sources" className="section">
        <div className="section-inner">
          <div className="section-eyebrow">Sources</div>
          <div className="section-body">
            <h2 className="section-heading">Key readings</h2>
            <p className="section-text section-text--dim">
              A curated reading list on AI, automation, and labor markets — coming soon.
            </p>
          </div>
        </div>
      </section>

      {/* ── Footer ──────────────────────────────── */}
      <footer className="footer">
        <span className="footer-copy">© 2025 Kseniia Oblasova</span>
        <div className="footer-links">
          <a href={GITHUB_URL} className="footer-link">Github</a>
          <a href={MEDIUM_URL} className="footer-link">Medium</a>
        </div>
      </footer>
    </>
  )
}
