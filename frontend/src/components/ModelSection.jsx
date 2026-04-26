import { useState } from 'react'

// ── Constants ──────────────────────────────────────────
const JOB_ZONES = [1, 2, 3, 4, 5]

const PCT_FEATURES = [
  { key: 'pct_computer', label: 'pct_computer', desc: 'How much of the job involves computing, software, or digital systems?' },
  { key: 'pct_physical', label: 'pct_physical', desc: 'How much of the job is hands-on or physical?' },
  { key: 'pct_communication', label: 'pct_communication', desc: 'How much of the job involves talking to people, teaching, or advising?' },
  { key: 'pct_analyze', label: 'pct_analyze', desc: 'How much of the job involves research, analysis, or problem-solving?' },
  { key: 'pct_manage', label: 'pct_manage', desc: 'How much of the job involves supervising, planning, or budgeting?' },
  { key: 'pct_creative', label: 'pct_creative', desc: 'How much of the job involves design, art, writing, or performance?' },
  { key: 'pct_textnative', label: 'pct_textnative', desc: 'How much of the job involves writing documents, coding, or data entry?' },
]

const DEFAULTS = {
  isBright: 0, isGreen: 0, JobZone: 3,
  MedianSalary: 55000,
  pct_computer: 30, pct_physical: 20, pct_communication: 40,
  pct_analyze: 35, pct_manage: 20, pct_creative: 10, pct_textnative: 30,
}

const TIMEOUT_MS = 20000

const INVALID_MSG = 'Sorry, invalid job title. Please try something else.'

const TITLE_BLOCKLIST = [
  "don't know", "dont know", "idk", "no idea", "not sure",
  "dunno", "n/a", "na", "none", "unknown", "test", "asdf", "qwerty",
  "hello", "hi", "hey", "yes", "no", "yehs", "yeh",
]

function validateJobTitle(title) {
  const t = title.trim().toLowerCase()
  if (t.length < 3) return false
  if (!/[a-zA-Z]/.test(t)) return false                   // no letters
  if (!/[aeiou]/i.test(t)) return false                   // no vowels
  if (/[bcdfghjklmnpqrstvwxyz]{5,}/i.test(t)) return false // 5+ consecutive consonants
  if (/^(.)\1+$/.test(t)) return false                    // all same character repeated
  if (TITLE_BLOCKLIST.includes(t)) return false
  return true
}

const fmt = (n) => '$' + n.toLocaleString('en-US')

// ── API helpers ────────────────────────────────────────
async function fetchWithTimeout(url, body) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    })

    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || `Request failed (${res.status})`)
    }

    return await res.json()
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('Request timed out — the server may be busy. Please try again.')
    throw e
  } finally {
    clearTimeout(timer)
  }
}

// ── Shared components ──────────────────────────────────
function ResultRow({ result }) {
  const isExposed = result.prediction === 1
  const rawConf = isExposed ? result.probability : 1 - result.probability
  const probability = Math.round(rawConf * 100)

  return (
    <div className="jt-result">
      <span className="result-label">AI exposed:</span>
      <span className={`result-yn${isExposed ? ' result-yn--yes' : ' result-yn--no'}`}>
        {isExposed ? 'Yes' : 'No'}
      </span>
      <span className="result-prob">
        with a probability of <strong>{probability}%</strong>
      </span>
    </div>
  )
}

function Disclaimer() {
  return (
    <p className="jt-disclaimer">
      This model provides a probabilistic estimate based on occupational data patterns,
      with an overall test-set accuracy of approximately 70%. It is not a definitive
      assessment of any individual's employment prospects.
    </p>
  )
}

// ── Job Title form ─────────────────────────────────────
function JobTitleForm() {
  const [jobTitle, setJobTitle] = useState('')
  const [jobDesc, setJobDesc] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!jobTitle.trim()) return
    setResult(null)
    setError(null)

    if (!validateJobTitle(jobTitle)) {
      setError(INVALID_MSG)
      return
    }

    setLoading(true)
    try {
      const data = await fetchWithTimeout('/api/predict/job-title', {
        job_title: jobTitle.trim(),
        job_description: jobDesc.trim() || null,
      })
      setResult(data)
    } catch (err) {
      setError(err.message === 'INVALID_JOB_TITLE' ? INVALID_MSG : err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="panel-content">
      <form className="jt-form" onSubmit={handleSubmit} noValidate>
        <div className="jt-fields">
          <div className="field">
            <label className="field-label" htmlFor="job-title">Job title</label>
            <input
              id="job-title" type="text" className="field-input"
              placeholder="e.g. Data Analyst"
              value={jobTitle} onChange={e => setJobTitle(e.target.value)}
              autoComplete="off"
            />
          </div>
          <div className="field">
            <label className="field-label" htmlFor="job-desc">
              Job description <span className="field-hint">(optional)</span>
            </label>
            <textarea
              id="job-desc" className="field-textarea" rows={4}
              placeholder="Paste a job posting or brief description…"
              value={jobDesc} onChange={e => setJobDesc(e.target.value)}
            />
          </div>
        </div>
        <button type="submit" className="submit-btn" disabled={!jobTitle.trim() || loading}>
          {loading ? 'Analysing…' : 'Predict exposure'}
        </button>
      </form>
      {error && <p className="result-error">{error}</p>}
      {result && <ResultRow result={result} />}
      <Disclaimer />
    </div>
  )
}

// ── Manual form ────────────────────────────────────────
function ManualForm() {
  const [f, setF] = useState(DEFAULTS)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const set = (key, val) => { setF(prev => ({ ...prev, [key]: val })); setResult(null); setError(null) }

  const salaryPct = ((f.MedianSalary - 10000) / (200000 - 10000)) * 100

  function handlePct(key, raw) {
    const n = raw === '' ? '' : Math.min(100, Math.max(0, Number(raw)))
    set(key, n)
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setResult(null)
    setError(null)

    try {
      // Send pct values as 0–100; backend divides by 100
      const data = await fetchWithTimeout('/api/predict/manual', {
        isBright: f.isBright,
        isGreen: f.isGreen,
        JobZone: f.JobZone,
        MedianSalary: f.MedianSalary,
        pct_computer: f.pct_computer,
        pct_physical: f.pct_physical,
        pct_communication: f.pct_communication,
        pct_analyze: f.pct_analyze,
        pct_manage: f.pct_manage,
        pct_creative: f.pct_creative,
        pct_textnative: f.pct_textnative,
      })
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="panel-content">
      <form className="manual-form" onSubmit={handleSubmit} noValidate>
        <div className="mf-top">
          <div className="mf-field">
            <span className="mf-label">isBright</span>
            <span className="mf-desc">Is this occupation having strong future job growth?</span>
            <div className="yn-group">
              <button type="button" className={`yn-btn${f.isBright === 1 ? ' yn-btn--on' : ''}`} onClick={() => set('isBright', 1)}>Yes</button>
              <button type="button" className={`yn-btn${f.isBright === 0 ? ' yn-btn--on' : ''}`} onClick={() => set('isBright', 0)}>No</button>
            </div>
          </div>

          <div className="mf-field">
            <span className="mf-label">isGreen</span>
            <span className="mf-desc">Is this occupation linked to the green/sustainable economy?</span>
            <div className="yn-group">
              <button type="button" className={`yn-btn${f.isGreen === 1 ? ' yn-btn--on' : ''}`} onClick={() => set('isGreen', 1)}>Yes</button>
              <button type="button" className={`yn-btn${f.isGreen === 0 ? ' yn-btn--on' : ''}`} onClick={() => set('isGreen', 0)}>No</button>
            </div>
          </div>

          <div className="mf-field">
            <span className="mf-label">JobZone</span>
            <span className="mf-desc">Education and experience level needed<br />(1 = minimal, 5 = graduate degree)</span>
            <div className="jz-group">
              {JOB_ZONES.map(n => (
                <button key={n} type="button"
                  className={`jz-btn${f.JobZone === n ? ' jz-btn--on' : ''}`}
                  onClick={() => set('JobZone', n)}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          <div className="mf-field">
            <span className="mf-label">Median Salary</span>
            <span className="mf-desc">Typical annual salary for this occupation (USD)</span>
            <div className="salary-wrap">
              <span className="salary-value">{fmt(f.MedianSalary)}</span>
              <input
                type="range" className="salary-range"
                min={10000} max={200000} step={1000}
                value={f.MedianSalary}
                style={{ background: `linear-gradient(to right, var(--blue) ${salaryPct}%, var(--border-mid) ${salaryPct}%)` }}
                onChange={e => set('MedianSalary', Number(e.target.value))}
              />
              <div className="salary-ticks">
                <span>$10,000</span>
                <span>$200,000</span>
              </div>
            </div>
          </div>
        </div>

        <div className="mf-pct-grid">
          {PCT_FEATURES.map(({ key, label, desc }) => (
            <div key={key} className="mf-field">
              <span className="mf-label">{label}</span>
              <span className="mf-desc">{desc}</span>
              <div className="pct-input-wrap">
                <input
                  type="number" className="pct-input"
                  min={0} max={100} step={1}
                  value={f[key]}
                  onChange={e => handlePct(key, e.target.value)}
                />
                <span className="pct-suffix">%</span>
              </div>
            </div>
          ))}
        </div>

        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? 'Predicting…' : 'Predict exposure'}
        </button>
      </form>

      {error && <p className="result-error">{error}</p>}
      {result && <ResultRow result={result} />}
      <Disclaimer />
    </div>
  )
}

// ── Root ───────────────────────────────────────────────
export default function ModelSection() {
  const [mode, setMode] = useState(null)

  const select = (m) => setMode(prev => prev === m ? null : m)

  return (
    <div className="model-wrap">
      <h2 className="model-heading">Try the model</h2>

      <p className="model-hint">
        <strong>Enter Job Title</strong> resolves your role via O*NET and predicts automatically.
        <br />
        <strong>Manual Mode</strong> lets you set each feature directly using sliders and toggles.
      </p>

      <div className="model-choices">
        <button
          className={`choice-btn choice-btn--pill${mode === 'job-title' ? ' choice-btn--active' : ''}`}
          onClick={() => select('job-title')}
        >
          Enter Job Title
        </button>
        <button
          className={`choice-btn choice-btn--rect${mode === 'manual' ? ' choice-btn--active' : ''}`}
          onClick={() => select('manual')}
        >
          Manual Mode
        </button>
      </div>

      {mode && (
        <div className="mode-panel-reveal" key={mode}>
          {mode === 'job-title' && <JobTitleForm />}
          {mode === 'manual' && <ManualForm />}
        </div>
      )}
    </div>
  )
}
