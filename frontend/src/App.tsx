import { ChangeEvent, FormEvent, useMemo, useRef, useState } from 'react'

type Severity = 'low' | 'medium' | 'high' | 'critical'

type ComplianceIssue = {
  id: string
  issue_type:
    | 'outdated_standard'
    | 'outdated_spec'
    | 'material_mismatch'
    | 'bom_mismatch'
    | 'missing_reference'
    | 'ocr_uncertain'
  severity: Severity
  message: string
  page: number | null
  section: string
  evidence: string
  expected_value: string | null
  found_value: string | null
  recommendation: string
}

type AnalysisResult = {
  run_id: string
  summary: string
  sections_detected: string[]
  issues: ComplianceIssue[]
  meta: {
    pages_processed: number
    used_foundational_context: boolean
    context_files_count: number
    inference_mode: 'online' | 'local'
    llm_enabled: boolean
    llm_used: boolean
    llm_model: string | null
    llm_endpoint: string | null
    llm_error: string | null
  }
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() || 'http://localhost:8000'

export function App() {
  const analyzerRef = useRef<HTMLElement | null>(null)
  const [drawing, setDrawing] = useState<File | null>(null)
  const [contextFiles, setContextFiles] = useState<File[]>([])
  const [useFoundational, setUseFoundational] = useState(false)
  const [inferenceMode, setInferenceMode] = useState<'online' | 'local'>('online')
  const [userApiKey, setUserApiKey] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const severeCount = useMemo(() => {
    if (!result) return 0
    return result.issues.filter((i) => i.severity === 'high' || i.severity === 'critical').length
  }, [result])

  const onDrawingChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null
    setDrawing(file)
  }

  const onContextChange = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? [])
    setContextFiles(files)
  }

  const scrollToAnalyzer = () => {
    analyzerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault()
    if (!drawing) {
      setError('Please upload one drawing first.')
      return
    }
    if (inferenceMode === 'online' && !userApiKey.trim()) {
      setError('Paste your Ollama API key to run in online mode.')
      return
    }

    setError(null)
    setIsLoading(true)
    setResult(null)

    const formData = new FormData()
    formData.append('drawing', drawing)
    formData.append('use_foundational_context', String(useFoundational))
    formData.append('inference_mode', inferenceMode)
    if (inferenceMode === 'online' && userApiKey.trim()) {
      formData.append('ollama_api_key', userApiKey.trim())
    }
    contextFiles.forEach((file) => formData.append('context_files', file))

    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const body = await response.text()
        throw new Error(body || 'Analysis failed')
      }

      const data = (await response.json()) as AnalysisResult
      setResult(data)
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : 'Unknown error')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="site-shell">
      <section className="hero-wrap">
        <div className="hero-inner">
          <h1>DraftWorks</h1>
          <p>
            Automatically validate engineering drawings against current standards and catch errors before they
            trigger costly rework and delays.
          </p>
          <button className="hero-btn" onClick={scrollToAnalyzer}>Try Sample</button>
        </div>
      </section>

      <section className="content-wrap split-block">
        <h2>Manual drawing review is slow and error prone</h2>
        <div className="stack-cards">
          <article>
            <h3>Explain the idea</h3>
            <p>State what changed, where it changed, and what should replace it.</p>
          </article>
          <article>
            <h3>In small parts</h3>
            <p>Break checks into standards, specs, materials, and BOM consistency.</p>
          </article>
          <article>
            <h3>With more detail</h3>
            <p>Attach evidence and expected values so reviewers trust each flagged issue.</p>
          </article>
        </div>
      </section>

      <section className="content-wrap panel-grid" ref={analyzerRef}>
        <article className="panel">
          <h3>Manual drawing review is slow and error prone</h3>
          <ul>
            <li>Engineers manually cross-check drawings against standards, specs, and BOMs.</li>
            <li>Small inconsistencies lead to costly review cycles and production delays.</li>
            <li>Updates across materials and parts are difficult to track.</li>
          </ul>
        </article>

        <article className="panel">
          <h3>How it works</h3>
          <ol>
            <li>Upload drawing (PDF/image)</li>
            <li>Upload context</li>
            <li>Compare and review issues</li>
          </ol>

          <form onSubmit={onSubmit} className="form-grid">
            <fieldset className="mode-switch" aria-label="Inference mode">
              <legend>Run mode</legend>
              <label className="inline">
                <input
                  type="radio"
                  name="inference_mode"
                  checked={inferenceMode === 'online'}
                  onChange={() => setInferenceMode('online')}
                />
                Run online
              </label>
              <label className="inline">
                <input
                  type="radio"
                  name="inference_mode"
                  checked={inferenceMode === 'local'}
                  onChange={() => setInferenceMode('local')}
                />
                Run locally on device
              </label>
            </fieldset>

            {inferenceMode === 'online' && (
              <label>
                Ollama API key
                <input
                  type="password"
                  placeholder="Paste your Ollama API key"
                  value={userApiKey}
                  onChange={(event) => setUserApiKey(event.target.value)}
                />
              </label>
            )}

            <label>
              Drawing file
              <input type="file" accept=".pdf,.jpg,.jpeg,.png,.webp,image/*,application/pdf" onChange={onDrawingChange} />
            </label>

            <label>
              Context files
              <input type="file" multiple onChange={onContextChange} />
            </label>

            <label className="inline checkbox-line">
              <input
                type="checkbox"
                checked={useFoundational}
                onChange={(event) => setUseFoundational(event.target.checked)}
              />
              Include foundational org context
            </label>

            <button type="submit" disabled={isLoading}>{isLoading ? 'Analyzing…' : 'Compare'}</button>
          </form>

          {error && <p className="error">{error}</p>}
        </article>

        <article className="panel">
          <h3>What DraftWorks does</h3>
          <ul>
            <li>Detect outdated standards and specification references.</li>
            <li>Validate material and coating compatibility.</li>
            <li>Flag required updates directly for engineering review.</li>
          </ul>
        </article>
      </section>

      {result && (
        <section className="content-wrap results-wrap">
          <h2>Results</h2>
          <p>{result.summary}</p>
          <div className="kpis">
            <span className="pill">Total issues: {result.issues.length}</span>
            <span className="pill">High/Critical: {severeCount}</span>
            <span className="pill">Sections: {result.sections_detected.join(', ') || 'none'}</span>
            <span className="pill">Mode: {result.meta.inference_mode}</span>
            <span className="pill">LLM: {result.meta.llm_used ? `used (${result.meta.llm_model ?? 'unknown'})` : 'not used'}</span>
          </div>
          {result.meta.llm_error && <p className="error">LLM fallback: {result.meta.llm_error}</p>}

          <table>
            <thead>
              <tr>
                <th>Type</th>
                <th>Severity</th>
                <th>Page</th>
                <th>Section</th>
                <th>Evidence</th>
                <th>Expected</th>
                <th>Found</th>
              </tr>
            </thead>
            <tbody>
              {result.issues.map((issue) => (
                <tr key={issue.id}>
                  <td>{issue.issue_type}</td>
                  <td>{issue.severity}</td>
                  <td>{issue.page ?? '-'}</td>
                  <td>{issue.section}</td>
                  <td>{issue.evidence}</td>
                  <td>{issue.expected_value ?? '-'}</td>
                  <td>{issue.found_value ?? '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

    </main>
  )
}
