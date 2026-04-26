import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from 'react'

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

const fileKey = (file: File) => `${file.name}:${file.size}:${file.lastModified}`

export function App() {
  const resultsRef = useRef<HTMLElement | null>(null)
  const [drawing, setDrawing] = useState<File | null>(null)
  const [contextFiles, setContextFiles] = useState<File[]>([])
  const [useFoundational, setUseFoundational] = useState(false)
  const [inferenceMode, setInferenceMode] = useState<'online' | 'local'>('online')
  const [inferenceOpen, setInferenceOpen] = useState(true)
  const [userApiKey, setUserApiKey] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const severeCount = useMemo(() => {
    if (!result) return 0
    return result.issues.filter((i) => i.severity === 'high' || i.severity === 'critical').length
  }, [result])

  useEffect(() => {
    if (!result) return
    resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [result])

  useEffect(() => {
    if (inferenceMode === 'online' && userApiKey.trim()) {
      setInferenceOpen(false)
    }
  }, [inferenceMode, userApiKey])

  const onDrawingChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null
    setDrawing(file)
  }

  const onContextChange = (event: ChangeEvent<HTMLInputElement>) => {
    const incoming = Array.from(event.target.files ?? [])
    if (!incoming.length) return

    setContextFiles((prev) => {
      const map = new Map(prev.map((f) => [fileKey(f), f]))
      incoming.forEach((file) => map.set(fileKey(file), file))
      return Array.from(map.values())
    })
    event.target.value = ''
  }

  const removeContextFile = (target: File) => {
    const targetKey = fileKey(target)
    setContextFiles((prev) => prev.filter((file) => fileKey(file) !== targetKey))
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
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">DraftWorks</p>
          <h1>Engineering Review Workbench</h1>
          <p className="subline">Upload drawing + context, run checks, and ship findings faster.</p>
        </div>
        <div className="status-chips">
          <span className="pill">Mode: {inferenceMode}</span>
          <span className="pill">Drawing: {drawing ? 'attached' : 'missing'}</span>
          <span className="pill">Context files: {contextFiles.length}</span>
        </div>
      </header>

      <section className="app-workspace">
        <aside className="control-panel">
          <h2>Run Analysis</h2>
          <p>Complete these steps, then hit compare.</p>

          <form onSubmit={onSubmit} className="form-grid">
            <section className="setting-card">
              <div className="card-head">
                <h3>1. Inference Settings</h3>
                <button
                  type="button"
                  className="ghost-btn"
                  onClick={() => setInferenceOpen((prev) => !prev)}
                >
                  {inferenceOpen ? 'Collapse' : 'Edit'}
                </button>
              </div>

              {inferenceOpen ? (
                <div className="stack">
                  <fieldset className="mode-switch" aria-label="Inference mode">
                    <legend>Inference mode</legend>
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
                      Ollama API key
                      <input
                        type="password"
                        placeholder="Paste your Ollama API key"
                        value={userApiKey}
                        onChange={(event) => setUserApiKey(event.target.value)}
                      />
                    </label>
                  )}
                </div>
              ) : (
                <p className="mini-note">
                  {inferenceMode === 'online' && userApiKey.trim()
                    ? 'Online mode selected. API key saved.'
                    : `Mode selected: ${inferenceMode}`}
                </p>
              )}
            </section>

            <section className="setting-card">
              <h3>2. Files</h3>

              <label>
                Drawing file
                <input
                  type="file"
                  accept=".pdf,.jpg,.jpeg,.png,.webp,image/*,application/pdf"
                  onChange={onDrawingChange}
                />
              </label>

              <label>
                Context files (multi-select supported)
                <input type="file" multiple onChange={onContextChange} />
              </label>

              {contextFiles.length > 0 && (
                <ul className="file-list">
                  {contextFiles.map((file) => (
                    <li key={fileKey(file)}>
                      <span>{file.name}</span>
                      <button type="button" className="chip-btn" onClick={() => removeContextFile(file)}>
                        Remove
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              <label className="inline checkbox-line">
                <input
                  type="checkbox"
                  checked={useFoundational}
                  onChange={(event) => setUseFoundational(event.target.checked)}
                />
                Include foundational org context (SurrealDB)
              </label>
            </section>

            <button className="primary-btn" type="submit" disabled={isLoading}>
              {isLoading ? 'Analyzing…' : 'Compare'}
            </button>
          </form>

          {error && <p className="error">{error}</p>}
        </aside>

        <section className="results-panel" ref={resultsRef}>
          <div className="results-head">
            <h2>Results</h2>
            {result ? <p>{result.summary}</p> : <p>Waiting for your first run.</p>}
          </div>

          {result && (
            <>
              <div className="kpis">
                <span className="pill">Total issues: {result.issues.length}</span>
                <span className="pill">High/Critical: {severeCount}</span>
                <span className="pill">Sections: {result.sections_detected.join(', ') || 'none'}</span>
                <span className="pill">LLM: {result.meta.llm_used ? `used (${result.meta.llm_model ?? 'unknown'})` : 'not used'}</span>
              </div>
              {result.meta.llm_error && <p className="error">LLM fallback: {result.meta.llm_error}</p>}

              <div className="table-wrap">
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
              </div>
            </>
          )}
        </section>
      </section>
    </main>
  )
}
