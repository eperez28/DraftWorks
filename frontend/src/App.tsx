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

type ZoneItemRow = {
  page: number
  zone: string
  object_key: string
  object_values: string[]
  line_number: number
}

type AnalysisResult = {
  run_id: string
  summary: string
  sections_detected: string[]
  issues: ComplianceIssue[]
  zone_rows: ZoneItemRow[]
  zone_markdown: string
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
type CloudKeySource = 'app' | 'byok'

export function App() {
  const resultsRef = useRef<HTMLElement | null>(null)
  const [drawing, setDrawing] = useState<File | null>(null)
  const [contextFiles, setContextFiles] = useState<File[]>([])
  const [useFoundational, setUseFoundational] = useState(true)
  const [inferenceMode, setInferenceMode] = useState<'online' | 'local'>('online')
  const [inferenceOpen, setInferenceOpen] = useState(false)
  const [cloudKeySource, setCloudKeySource] = useState<CloudKeySource>('app')
  const [userApiKey, setUserApiKey] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const severeCount = useMemo(() => {
    if (!result) return 0
    return result.issues.filter((i) => i.severity === 'high' || i.severity === 'critical').length
  }, [result])

  const zoneCountByName = useMemo(() => {
    if (!result) return new Map<string, number>()
    const counts = new Map<string, number>()
    result.zone_rows.forEach((row) => counts.set(row.zone, (counts.get(row.zone) ?? 0) + 1))
    return counts
  }, [result])

  useEffect(() => {
    if (!result) return
    resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [result])

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
    if (inferenceMode === 'online' && cloudKeySource === 'byok' && !userApiKey.trim()) {
      setError('Paste your Ollama API key to run in BYOK mode.')
      return
    }

    setError(null)
    setIsLoading(true)
    setResult(null)

    const formData = new FormData()
    formData.append('drawing', drawing)
    formData.append('use_foundational_context', String(useFoundational))
    formData.append('inference_mode', inferenceMode)
    if (inferenceMode === 'online' && cloudKeySource === 'byok' && userApiKey.trim()) {
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
          <div className="panel-head">
            <div>
              <h2>Run Analysis</h2>
              <p>Upload drawing + context, then hit compare.</p>
            </div>
            <button type="button" className="settings-btn" onClick={() => setInferenceOpen((prev) => !prev)}>
              <span aria-hidden="true">⚙</span>
              Settings
            </button>
          </div>

          <form onSubmit={onSubmit} className="form-grid">
            <section className="setting-card">
              <h3>1. Files</h3>

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

            </section>

            {inferenceOpen && (
              <section className="setting-card settings-card">
                <h3>Settings (Optional)</h3>
                <div className="stack">
                  <label className="inline checkbox-line">
                    <input
                      type="checkbox"
                      checked={useFoundational}
                      onChange={(event) => setUseFoundational(event.target.checked)}
                    />
                    Include foundational org context (SurrealDB)
                  </label>

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
                    <>
                      <fieldset className="mode-switch" aria-label="Cloud API key source">
                        <legend>Cloud API key source</legend>
                        <label className="inline">
                          <input
                            type="radio"
                            name="cloud_key_source"
                            checked={cloudKeySource === 'app'}
                            onChange={() => setCloudKeySource('app')}
                          />
                          Use app key (default)
                        </label>
                        <label className="inline">
                          <input
                            type="radio"
                            name="cloud_key_source"
                            checked={cloudKeySource === 'byok'}
                            onChange={() => setCloudKeySource('byok')}
                          />
                          Bring your own key (BYOK)
                        </label>
                      </fieldset>

                      {cloudKeySource === 'byok' && (
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
                    </>
                  )}
                </div>
              </section>
            )}

            <button className="primary-btn" type="submit" disabled={isLoading}>
              {isLoading ? 'Analyzing…' : 'Compare'}
            </button>

            <p className="mini-note settings-summary">
              {inferenceMode === 'online'
                ? cloudKeySource === 'app'
                  ? 'Online mode: app-managed key.'
                  : userApiKey.trim()
                    ? 'Online mode: BYOK enabled.'
                    : 'Online mode: BYOK selected, key missing.'
                : 'Local mode: on-device Ollama.'}
            </p>
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
                <span className="pill">Zone rows: {result.zone_rows.length}</span>
              </div>
              {result.meta.llm_error && <p className="error">LLM fallback: {result.meta.llm_error}</p>}

              <div className="kpis">
                <span className="pill">notes: {zoneCountByName.get('notes') ?? 0}</span>
                <span className="pill">revision: {zoneCountByName.get('revision_block') ?? 0}</span>
                <span className="pill">title: {zoneCountByName.get('title_block') ?? 0}</span>
                <span className="pill">drawing: {zoneCountByName.get('drawing_area') ?? 0}</span>
              </div>

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

              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Page</th>
                      <th>Zone</th>
                      <th>Object key</th>
                      <th>Values</th>
                      <th>Line</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.zone_rows.map((row, idx) => (
                      <tr key={`${row.page}-${row.zone}-${row.object_key}-${idx}`}>
                        <td>{row.page}</td>
                        <td>{row.zone}</td>
                        <td>{row.object_key}</td>
                        <td>{row.object_values.join(' | ')}</td>
                        <td>{row.line_number}</td>
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
