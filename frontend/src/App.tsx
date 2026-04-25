import { ChangeEvent, FormEvent, useMemo, useState } from 'react'

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

export function App() {
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

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault()
    if (!drawing) {
      setError('Please upload one drawing PDF first.')
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
      const response = await fetch('http://localhost:8000/api/analyze', {
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
    <main className="page">
      <header className="hero">
        <h1>DraftWorks</h1>
        <p className="subhead">AI drawing compliance checker MVP for defense and mechanical workflows.</p>
      </header>

      <section className="card">
        <h2>Upload and Analyze</h2>
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
              Run online (Ollama Cloud)
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
              Ollama API key (required for online mode)
              <input
                type="password"
                placeholder="Paste your Ollama API key"
                value={userApiKey}
                onChange={(event) => setUserApiKey(event.target.value)}
              />
              <small>Your key is sent only with this request and is not persisted by the app.</small>
            </label>
          )}

          <label>
            Drawing PDF (required)
            <input type="file" accept="application/pdf" onChange={onDrawingChange} />
          </label>

          <label>
            Context files (optional: CSV, TXT, JSON)
            <input type="file" multiple onChange={onContextChange} />
          </label>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={useFoundational}
              onChange={(event) => setUseFoundational(event.target.checked)}
            />
            <span>
              Include foundational org context (SurrealDB when connected)
              <span
                className="tooltip"
                title="Planned rule: drawing-view item callouts will be validated against BOM rows for existence and quantity mismatches."
              >
                i
              </span>
            </span>
          </label>

          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Analyzing…' : 'Run compliance check'}
          </button>
        </form>
        <div className="instructions">
          <h3>How to run</h3>
          <ol>
            <li>
              <strong>Run online:</strong> create key at <code>https://ollama.com/settings/keys</code>, select online mode, paste key, upload files, run.
            </li>
            <li>
              <strong>Run locally on device:</strong> start Ollama locally, run <code>ollama pull gemma4:e4b</code>, choose local mode, upload files, run.
            </li>
          </ol>
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      {result && (
        <section className="card">
          <h2>Results</h2>
          <p>{result.summary}</p>
          <div className="kpis">
            <span className="pill">Total issues: {result.issues.length}</span>
            <span className="pill">High/Critical: {severeCount}</span>
            <span className="pill">Sections: {result.sections_detected.join(', ') || 'none'}</span>
            <span className="pill">Mode: {result.meta.inference_mode}</span>
            <span className="pill">LLM: {result.meta.llm_used ? `used (${result.meta.llm_model ?? 'unknown'})` : 'not used'}</span>
          </div>
          {result.meta.llm_endpoint && <p className="meta-line">Endpoint: <code>{result.meta.llm_endpoint}</code></p>}
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
