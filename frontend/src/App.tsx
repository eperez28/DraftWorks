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

type ComparisonRow = {
  sheet: number
  zone: string
  existing_text: string
  replace_with: string
  change_type: string
  source_basis: string
  priority: string
  notes: string
}

type RawComparisonRow = Partial<ComparisonRow> & {
  page?: number
  found_value?: string
  suggested_value?: string | null
  status?: string
}

type AnalysisResult = {
  run_id: string
  summary: string
  sections_detected: string[]
  issues: ComplianceIssue[]
  zone_rows: ZoneItemRow[]
  comparison_rows: ComparisonRow[]
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
    foundational_context_error: string | null
  }
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() || 'http://localhost:8000'

const fileKey = (file: File) => `${file.name}:${file.size}:${file.lastModified}`
type CloudKeySource = 'app' | 'byok'

export function App() {
  const resultsRef = useRef<HTMLElement | null>(null)
  const contextInputRef = useRef<HTMLInputElement | null>(null)
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

  const openContextPicker = () => {
    contextInputRef.current?.click()
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
      const normalizedRows = (data.comparison_rows as RawComparisonRow[]).map((row) => normalizeComparisonRow(row))
      setResult({ ...data, comparison_rows: normalizedRows })
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : 'Unknown error')
    } finally {
      setIsLoading(false)
    }
  }

  const normalizeComparisonRow = (row: RawComparisonRow): ComparisonRow => {
    const existing = String(row.existing_text ?? row.found_value ?? '').trim()
    const replace = String(row.replace_with ?? row.suggested_value ?? '').trim()
    const status = String(row.status ?? '').trim().toLowerCase()
    const isLikelyChange = !!replace && replace !== '-' && !['no change', 'no change recommended'].includes(replace.toLowerCase())

    return {
      sheet: Number(row.sheet ?? row.page ?? 0) || 0,
      zone: String(row.zone ?? '').trim(),
      existing_text: existing,
      replace_with: replace || (status === 'no_context' ? 'NO CHANGE RECOMMENDED' : 'NO CHANGE'),
      change_type: String(row.change_type ?? (isLikelyChange ? 'Value Update' : 'Review Only')).trim(),
      source_basis: String(row.source_basis ?? (status === 'no_context' ? 'No governing source found' : 'Context comparison')).trim(),
      priority: String(row.priority ?? (isLikelyChange ? 'High' : '—')).trim(),
      notes: String(row.notes ?? '').trim(),
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
                <div className="file-picker">
                  <button type="button" className="file-picker-btn" onClick={openContextPicker}>
                    Choose Files
                  </button>
                  <span className="file-picker-status">
                    {contextFiles.length > 0
                      ? `${contextFiles.length} file${contextFiles.length === 1 ? '' : 's'} selected`
                      : 'No files selected'}
                  </span>
                  <input
                    ref={contextInputRef}
                    className="visually-hidden-input"
                    type="file"
                    multiple
                    hidden
                    onChange={onContextChange}
                  />
                </div>
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
              </div>
              {result.meta.llm_error && <p className="error">LLM fallback: {result.meta.llm_error}</p>}
              {result.meta.foundational_context_error && (
                <p className="error">Foundational context fallback: {result.meta.foundational_context_error}</p>
              )}

              {result.issues.length > 0 && (
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
              )}
              {result.issues.length === 0 && result.comparison_rows.length === 0 && (
                <p className="mini-note">No issues or comparison rows were detected from this run.</p>
              )}

              {result.comparison_rows.length > 0 && (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Sheet</th>
                        <th>Zone</th>
                        <th>Existing Text / Field</th>
                        <th>Replace With</th>
                        <th>Change Type</th>
                        <th>Source / Basis</th>
                        <th>Priority</th>
                        <th>Notes</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.comparison_rows.map((row, idx) => (
                        <tr key={`${row.sheet}-${row.zone}-${idx}`}>
                          <td>{row.sheet}</td>
                          <td>{row.zone}</td>
                          <td>{row.existing_text || '-'}</td>
                          <td>{row.replace_with || '-'}</td>
                          <td>{row.change_type || '-'}</td>
                          <td>{row.source_basis || '-'}</td>
                          <td>{row.priority || '-'}</td>
                          <td>{row.notes || '-'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </section>
      </section>
    </main>
  )
}
