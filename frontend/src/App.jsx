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
            <span className="hero-title-main">AI and the Job Market: </span>
            <span className="hero-subtitle">A Machine Learning Approach<br /> to Predicting Occupational<br /> AI Exposure</span>
          </h1>

          <p className="hero-desc">
            Enter a job title. Find out whether an occupation is exposed to AI — powered by O*NET task data,
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
              This tool estimates an occupation's AI exposure — whether any of its tasks involve
              work that AI systems can both plausibly perform in principle and have actually been
              observed performing in practice. Exposure is not displacement: a positive prediction
              means parts of the job overlap with what AI can do, not that the role is at risk of disappearing.
              It maps job titles to O*NET standard occupational codes via Claude,
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
              A reading list on AI, automation, and labor markets used in this project.
            </p>
            <ol className="references">
              <li>Adam K. <em>onet-dataviz</em>. GitHub repository. <a href="https://github.com/adamkq/onet-dataviz" target="_blank" rel="noopener">https://github.com/adamkq/onet-dataviz</a></li>
              <li>Alderucci, D., Branstetter, L., Hovy, E., Runge, A., &amp; Zolas, N. (2020). Quantifying the impact of AI on productivity and labor demand: Evidence from U.S. Census microdata. Working paper. <a href="https://conference.nber.org/conf_papers/f204793.pdf" target="_blank" rel="noopener">https://conference.nber.org/conf_papers/f204793.pdf</a></li>
              <li>Anthropic (2026). <em>Anthropic Economic Index</em>. Hugging Face dataset. <a href="https://huggingface.co/datasets/Anthropic/EconomicIndex" target="_blank" rel="noopener">https://huggingface.co/datasets/Anthropic/EconomicIndex</a></li>
              <li>Anthropic (2026). <em>Economic Index — March 2026 report</em>. <a href="https://www.anthropic.com/research/economic-index-march-2026-report" target="_blank" rel="noopener">https://www.anthropic.com/research/economic-index-march-2026-report</a></li>
              <li>Backlinko (2025). <em>Claude users: usage statistics</em>. <a href="https://backlinko.com/claude-users" target="_blank" rel="noopener">https://backlinko.com/claude-users</a></li>
              <li>Colombo, E., Mercorio, F., Mezzanzanica, M., &amp; Vegetti, F. (2024). Mapping AI exposure across European labor markets. <em>Proceedings of IJCAI 2025</em>. <a href="https://www.ijcai.org/proceedings/2025/1066.pdf" target="_blank" rel="noopener">https://www.ijcai.org/proceedings/2025/1066.pdf</a></li>
              <li>Galeano, L. et al. (2025). By degrees: measuring employer demand for AI skills by educational requirements. <em>Workforce Currents</em>, Federal Reserve Bank of Atlanta. <a href="https://www.atlantafed.org/research-and-data/publications/workforce-currents/2025/05/21/01" target="_blank" rel="noopener">https://www.atlantafed.org/research-and-data/publications/workforce-currents/2025/05/21/01</a></li>
              <li>Handa, K. et al. (2025). How users interact with Claude across occupational tasks: evidence from the Anthropic Economic Index. <em>Anthropic Research</em>.</li>
              <li>Massenkoff, M. &amp; McCrory, P. (2025). Labor market impacts of AI: a new measure and early evidence. <em>Anthropic Research</em>. <a href="https://www.anthropic.com/research/labor-market-impacts" target="_blank" rel="noopener">https://www.anthropic.com/research/labor-market-impacts</a></li>
              <li>Parteka, A. &amp; Kordalska, A. (2023). Artificial intelligence and productivity: global evidence from AI patent and bibliometric data. <em>Technovation</em>, 125. <a href="https://doi.org/10.1016/j.technovation.2023.102764" target="_blank" rel="noopener">https://doi.org/10.1016/j.technovation.2023.102764</a></li>
              <li>Tamkin, A. et al. (2024). Clio: privacy-preserving insights into real-world AI use. <a href="https://arxiv.org/abs/2412.13678" target="_blank" rel="noopener">https://arxiv.org/abs/2412.13678</a></li>
              <li>U.S. Bureau of Labor Statistics. <em>Occupational Employment and Wage Statistics (OEWS) API</em>. <a href="https://www.bls.gov/oes/" target="_blank" rel="noopener">https://www.bls.gov/oes/</a></li>
              <li>U.S. Department of Labor. <em>O*NET Web Services</em>. <a href="https://services.onetcenter.org/" target="_blank" rel="noopener">https://services.onetcenter.org/</a></li>
              <li>Xie, M. &amp; Yan, B. (2024). Generative AI and customer-service productivity: evidence from a large-scale field deployment. <em>International Review of Economics &amp; Finance</em>. <a href="https://doi.org/10.1016/j.iref.2024.103408" target="_blank" rel="noopener">https://doi.org/10.1016/j.iref.2024.103408</a></li>
            </ol>
          </div>
        </div>
      </section>

      {/* ── Footer ──────────────────────────────── */}
      <footer className="footer">
        <span className="footer-copy">Kseniia Oblasova</span>
        <div className="footer-links">
          <a href={GITHUB_URL} className="footer-link">Github</a>
          <a href={MEDIUM_URL} className="footer-link">Medium</a>
        </div>
      </footer>
    </>
  )
}
