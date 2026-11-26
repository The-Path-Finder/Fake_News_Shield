import { useEffect, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

const SAMPLES = [
  {
    label: 'Reuters (Real)',
    url: 'https://www.reuters.com',
    text: 'WASHINGTON (Reuters) - The U.S. Federal Reserve held interest rates steady on Wednesday while signaling potential adjustments later this year based on inflation and labor market data. Policymakers noted that economic activity has continued to expand at a solid pace.',
  },
  {
    label: 'Viral Hoax (Fake)',
    url: '',
    text: 'BREAKING SHOCKING: Secret alien base discovered under the Eiffel Tower and world leaders have been hiding the truth from citizens for decades! Share this immediately before the mainstream media deletes it!',
  },
  {
    label: 'Hindi (Multilingual)',
    url: '',
    text: 'नई दिल्ली: भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) ने आज एक नए संचार उपग्रह के सफल प्रक्षेपण की घोषणा की। वैज्ञानिकों के अनुसार यह मिशन देश में डिजिटल कनेक्टिविटी को मजबूत करेगा।',
  },
]

export default function App() {
  const [tab, setTab] = useState('desk')
  const [text, setText] = useState('')
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const [status, setStatus] = useState({ online: false, info: 'Checking API...' })

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error('Offline'))))
      .then((data) => {
        setStatus({
          online: Boolean(data.model_loaded),
          info: `${data.model_type} · ${data.vocab_size?.toLocaleString()} features`,
        })
      })
      .catch(() => {
        setStatus({ online: false, info: 'API Offline' })
      })
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!text.trim()) {
      setError('Enter or paste a news article to run verification.')
      return
    }
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text.trim(),
          url: url.trim() || null,
          source: 'web',
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.detail || 'Prediction request failed.')
      }
      setResult(data)
      setHistory((prev) => [
        {
          id: Date.now(),
          snippet: text.trim().slice(0, 100) + (text.trim().length > 100 ? '...' : ''),
          label: data.label,
          confidence: data.confidence,
          language: data.language_name,
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
        ...prev.slice(0, 9),
      ])
    } catch (err) {
      setError(err.message || 'Could not reach the FastAPI backend.')
    } finally {
      setLoading(false)
    }
  }

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0
  const confidencePct = result ? Math.round(result.confidence * 100) : 0
  const meterColor =
    result?.risk_level === 'low'
      ? '#166534'
      : result?.risk_level === 'high'
        ? '#991b1b'
        : '#92400e'

  return (
    <div className="page">
      <header className="masthead">
        <div>
          <div className="masthead-kicker">AIP303 · Hallucination Handlers</div>
          <h1 className="masthead-title">Fake News Shield</h1>
        </div>
        <div className="masthead-meta">
          <span className="status-indicator">
            <span className={`status-dot ${status.online ? 'online' : 'offline'}`} />
            {status.online ? 'FastAPI Connected' : 'FastAPI Offline'}
          </span>
          <span>{status.info}</span>
        </div>
      </header>

      <nav className="subnav">
        <button
          type="button"
          className={`subnav-btn ${tab === 'desk' ? 'active' : ''}`}
          onClick={() => setTab('desk')}
        >
          Verification Desk
        </button>
        <button
          type="button"
          className={`subnav-btn ${tab === 'history' ? 'active' : ''}`}
          onClick={() => setTab('history')}
        >
          Log ({history.length})
        </button>
        <button
          type="button"
          className={`subnav-btn ${tab === 'system' ? 'active' : ''}`}
          onClick={() => setTab('system')}
        >
          Pipeline &amp; Architecture
        </button>
      </nav>

      {tab === 'desk' && (
        <main className="workspace">
          <section className="panel">
            <div className="panel-header">
              <h2 className="panel-title">Article Input</h2>
              <div className="samples">
                {SAMPLES.map((s) => (
                  <button
                    key={s.label}
                    type="button"
                    className="sample-link"
                    onClick={() => {
                      setText(s.text)
                      setUrl(s.url)
                      setError('')
                    }}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>

            <form onSubmit={handleSubmit}>
              <label htmlFor="article-url" className="field-label">
                Source URL (optional)
              </label>
              <input
                id="article-url"
                type="url"
                className="url-input"
                placeholder="https://..."
                value={url}
                onChange={(e) => setUrl(e.target.value)}
              />

              <label htmlFor="article-text" className="field-label">
                Article Text (English or Hindi)
              </label>
              <textarea
                id="article-text"
                className="article-input"
                placeholder="Paste the headline and body of the news article..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />

              <div className="composer-footer">
                <span className="count-label">
                  {wordCount} words / {text.length} chars
                </span>
                <div className="actions">
                  <button
                    type="button"
                    className="btn-ghost"
                    onClick={() => {
                      setText('')
                      setUrl('')
                      setResult(null)
                      setError('')
                    }}
                  >
                    Clear
                  </button>
                  <button type="submit" className="btn-submit" disabled={loading}>
                    {loading ? 'Analyzing...' : 'Run Classification'}
                  </button>
                </div>
              </div>
            </form>

            {error && <div className="error-msg">{error}</div>}

            <div className="trace">
              <div className={`trace-item ${text.trim() ? 'active' : ''}`}>
                <span className="trace-step">01. INPUT</span>
                Raw article text
              </div>
              <div className={`trace-item ${result ? 'active' : ''}`}>
                <span className="trace-step">02. SPACY NLP</span>
                {result ? `${result.language_name} cleaned` : 'Detect & lemmatize'}
              </div>
              <div className={`trace-item ${result ? 'active' : ''}`}>
                <span className="trace-step">03. LINEAR SVM</span>
                TF-IDF (1–2 grams)
              </div>
              <div className={`trace-item ${result ? 'active' : ''}`}>
                <span className="trace-step">04. THRESHOLD</span>
                {result ? `${confidencePct}% confidence` : 'Confidence check'}
              </div>
            </div>
          </section>

          <aside className="panel">
            <div className="panel-header">
              <h2 className="panel-title">Assessment</h2>
            </div>

            {!result ? (
              <div className="empty-dossier">
                Submit an article on the left to inspect its language detection, TF-IDF feature
                overlap, and Linear SVM classification.
              </div>
            ) : (
              <div>
                <div className={`verdict-box ${result.risk_level}`}>
                  <div className="verdict-kicker">Classifier Verdict</div>
                  <div className="verdict-title">{result.label}</div>
                  <p className="verdict-rec">{result.recommendation}</p>
                  <div className="meter-row">
                    <div className="meter-header">
                      <span>Confidence</span>
                      <span>{confidencePct}%</span>
                    </div>
                    <div className="meter-track">
                      <div
                        className="meter-fill"
                        style={{ width: `${confidencePct}%`, backgroundColor: meterColor }}
                      />
                    </div>
                  </div>
                </div>

                <table className="data-table">
                  <tbody>
                    <tr>
                      <th>Detected Language</th>
                      <td>
                        {result.language_name} ({result.language})
                      </td>
                    </tr>
                    <tr>
                      <th>Matched TF-IDF Features</th>
                      <td>{result.matched_features}</td>
                    </tr>
                    <tr>
                      <th>Processed Tokens</th>
                      <td>{result.token_count}</td>
                    </tr>
                    <tr>
                      <th>SVM Decision Margin</th>
                      <td>{result.raw_score !== null ? result.raw_score : 'N/A'}</td>
                    </tr>
                  </tbody>
                </table>

                <div className="sources-section">
                  <span className="field-label">Cross-Check Trusted Sources</span>
                  <div className="sources-links">
                    {result.trusted_sources.map((src) => (
                      <a
                        key={src.name}
                        href={src.url}
                        target="_blank"
                        rel="noreferrer"
                        className="source-link"
                      >
                        {src.name}
                      </a>
                    ))}
                  </div>
                </div>

                <p className="disclaimer">{result.disclaimer}</p>
              </div>
            )}
          </aside>
        </main>
      )}

      {tab === 'history' && (
        <section className="panel">
          <div className="panel-header">
            <h2 className="panel-title">Session Log</h2>
          </div>
          {history.length === 0 ? (
            <p className="empty-dossier">No articles classified in this session.</p>
          ) : (
            history.map((item) => (
              <div key={item.id} className="history-row">
                <div>
                  <div className="history-snippet">{item.snippet}</div>
                  <div className="history-sub">
                    {item.language} · {item.time}
                  </div>
                </div>
                <span className="history-tag">
                  {item.label} ({Math.round(item.confidence * 100)}%)
                </span>
              </div>
            ))
          )}
        </section>
      )}

      {tab === 'system' && (
        <section className="panel">
          <div className="panel-header">
            <h2 className="panel-title">System Architecture</h2>
          </div>
          <div className="arch-list">
            <div className="arch-item">
              <h3>1. spaCy Preprocessing</h3>
              <p>
                Language detection combined with spaCy tokenization, lowercasing, stop-word removal
                (preserving negations), and lemmatization.
              </p>
            </div>
            <div className="arch-item">
              <h3>2. TF-IDF &amp; Linear SVM</h3>
              <p>
                Unigram and bigram TF-IDF vectorization (10,000 features) paired with a LinearSVC
                classifier selected over Logistic Regression and Multinomial Naive Bayes.
              </p>
            </div>
            <div className="arch-item">
              <h3>3. FastAPI /predict Endpoint</h3>
              <p>
                Scores incoming text from both the web app and browser extension, applying a
                confidence threshold to flag uncertain or out-of-vocabulary inputs.
              </p>
            </div>
            <div className="arch-item">
              <h3>4. Browser Extension</h3>
              <p>
                Manifest V3 extension that extracts article paragraphs from the active tab on click
                and displays the verdict with a fact-check disclaimer.
              </p>
            </div>
          </div>
        </section>
      )}
    </div>
  )
}
