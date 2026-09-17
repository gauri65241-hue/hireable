import { useState, useRef, useCallback, useEffect } from 'react'
import { useStore } from '@/store'
import { getToken, api } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { ScoreRing } from '@/components/ui/ScoreRing'
import { ScoreBadge } from '@/components/ui/ScoreBadge'
import { avatarHue, initials } from '@/lib/utils'
import { Play, ChevronDown } from 'lucide-react'

function PageHeader({ title, sub }: { title: string; sub: string }) {
  return (
    <div className="relative pb-[18px] mb-6" style={{ borderBottom: '1px solid var(--border)' }}>
      <div className="absolute bottom-[-1px] left-0 w-10 h-0.5 rounded-sm bg-[var(--accent)]" />
      <h1 className="font-display text-[20px] font-black tracking-[-0.5px] mb-1">{title}</h1>
      <p className="text-[13px] text-[var(--text2)]">{sub}</p>
    </div>
  )
}

type JudgeState = 'pending' | 'running' | 'done' | 'failed'

interface JudgeResult {
  state: JudgeState
  score: number
  elapsed: number
  reason?: string
}

interface AnalysisResult {
  matchScore: number
  confidence: number
  totalElapsed: number
  atsScore?: number
  semanticScore?: number
  missingSkills?: string[]
  judges: Record<string, JudgeResult>
}

const JUDGES = [
  { id: 'skills',     label: 'Skills',     weight: '35%', desc: 'Required & optional skills depth' },
  { id: 'experience', label: 'Experience', weight: '30%', desc: 'Years, projects, company relevance' },
  { id: 'seniority',  label: 'Seniority',  weight: '10%', desc: 'Career level determination' },
  { id: 'relevancy',  label: 'Relevancy',  weight: '15%', desc: 'Role alignment & domain fit' },
  { id: 'impact',     label: 'Impact',     weight: '10%', desc: 'Metrics, ownership, leadership' },
]

function JudgeCard({ judge, result }: { judge: typeof JUDGES[0]; result: JudgeResult }) {
  const isRunning = result.state === 'running'
  const isDone    = result.state === 'done'
  const isPending = result.state === 'pending'
  const isFailed  = result.state === 'failed'

  const borderColor = isDone ? (result.score >= 70 ? 'var(--green-border)' : result.score >= 40 ? 'var(--amber-border)' : 'var(--red-border)')
    : isRunning ? 'var(--accent-border)'
    : 'var(--border)'

  const bgColor = isDone ? (result.score >= 70 ? 'var(--green-bg)' : result.score >= 40 ? 'var(--amber-bg)' : 'var(--red-bg)')
    : isRunning ? 'var(--accent-bg)'
    : 'transparent'

  const scoreColor = result.score >= 70 ? 'var(--green)' : result.score >= 40 ? 'var(--amber)' : 'var(--red)'
  const barPct = isDone ? result.score : isRunning ? 45 : 0

  return (
    <div className="p-3.5 rounded-[var(--radius)] transition-all duration-300"
      style={{ border: `1px solid ${borderColor}`, background: bgColor, opacity: isPending ? 0.45 : 1 }}>
      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[12.5px] font-bold text-[var(--text)]">{judge.label}</span>
            <span className="text-[10px] font-bold px-1.5 py-0.5 rounded" style={{ background: 'var(--surface3)', color: 'var(--text3)', border: '1px solid var(--border)' }}>{judge.weight}</span>
          </div>
          <div className="text-[10.5px] text-[var(--text3)] mt-0.5">{judge.desc}</div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          {isRunning && (
            <div className="w-4 h-4 rounded-full border-2 border-t-[var(--accent)] border-[rgba(99,102,241,.2)]" style={{ animation: 'spin .7s linear infinite' }} />
          )}
          {isDone && <span className="font-display font-black text-[20px]" style={{ color: scoreColor }}>{result.score}</span>}
          {isFailed && <span className="text-[11px] text-[var(--red)]">Failed</span>}
          {isPending && <span className="text-[11px] text-[var(--text3)]">Waiting</span>}
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-1 rounded-full overflow-hidden" style={{ background: 'var(--surface3)' }}>
        <div className="h-full rounded-full transition-all duration-500"
          style={{ width: `${barPct}%`, background: isDone ? scoreColor : isRunning ? 'var(--accent)' : 'transparent', opacity: isRunning ? 0.6 : 1 }} />
      </div>

      {isDone && result.elapsed > 0 && (
        <div className="text-[10px] text-[var(--text3)] mt-1.5 font-mono">{result.elapsed.toFixed(1)}s</div>
      )}
    </div>
  )
}

export function Analyzer() {
  const { resumes, jobs, selectedResumes, selectedJobs, toggleResume, toggleJob } = useStore()
  const [model, setModel] = useState('')
  const [availableModels, setAvailableModels] = useState<{ group: string; models: string[] }[]>([])
  const [useCoach, setUseCoach] = useState(false)
  const [running, setRunning] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [status, setStatus] = useState('')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [judgeStates, setJudgeStates] = useState<Record<string, JudgeResult>>(() =>
    Object.fromEntries(JUDGES.map(j => [j.id, { state: 'pending' as JudgeState, score: 0, elapsed: 0 }]))
  )
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number; label: string } | null>(null)
  const [batchResults, setBatchResults] = useState<Array<{ resumeName: string; jobTitle: string; company: string; matchScore: number; atsScore: number; semanticScore: number; missingSkills: string[] }>>([])

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const startRef = useRef<number>(0)
  const abortRef = useRef<AbortController | null>(null)

  const resetJudges = () =>
    setJudgeStates(Object.fromEntries(JUDGES.map(j => [j.id, { state: 'pending' as JudgeState, score: 0, elapsed: 0 }])))

  const startTimer = () => {
    startRef.current = Date.now()
    timerRef.current = setInterval(() => setElapsed(Math.floor((Date.now() - startRef.current) / 1000)), 500)
  }
  const stopTimer = () => { if (timerRef.current) clearInterval(timerRef.current) }

  const streamOne = async (
    resume: { name: string; text: string },
    job: { title: string; company: string; description: string },
    signal: AbortSignal,
    onComplete: (r: AnalysisResult) => void,
  ) => {
    const token = getToken()
    const res = await fetch('/analyze/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      body: JSON.stringify({ resume_text: resume.text, job_description: job.description, use_coach: useCoach, llm_models: [model], candidate_name: resume.name, job_title: job.title }),
      signal,
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    if (!res.body) throw new Error('No response body')

    const reader = res.body.getReader()
    const dec = new TextDecoder()
    let buf = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      const parts = buf.split('\n\n')
      buf = parts.pop() ?? ''
      for (const part of parts) {
        const line = part.split('\n').find(l => l.startsWith('data: '))
        if (!line) continue
        try {
          const ev = JSON.parse(line.slice(6)) as Record<string, unknown>
          const event = ev.event as string
          if (event === 'ats') {
            const d = ev.data as Record<string, unknown>
            setStatus(`ATS — score ${d?.ats_score ?? '?'}`)
          } else if (event === 'judge_start') {
            const judge = ev.judge as string
            setJudgeStates(s => ({ ...s, [judge]: { state: 'running', score: 0, elapsed: 0 } }))
            setStatus(`Running ${judge} judge…`)
          } else if (event === 'judge') {
            const judge = ev.judge as string
            const score = Math.round((ev.score as number) ?? 0)
            const elap  = (ev.elapsed_s as number) ?? 0
            const ok    = ev.success as boolean
            setJudgeStates(s => ({ ...s, [judge]: { state: ok ? 'done' : 'failed', score, elapsed: elap } }))
          } else if (event === 'complete') {
            const matchScore  = Math.round((ev.match_score as number) ?? 0)
            const conf        = Math.round(((ev.confidence as number) ?? 0) * 100)
            const totalEl     = (ev.total_elapsed_s as number) ?? 0
            const missing     = (ev.missing_skills as string[]) ?? []
            const atsScore    = Math.round((ev.ats_score as number) ?? 0)
            const semScore    = Math.round(((ev.semantic_similarity as number) ?? 0) * 100)
            const r: AnalysisResult = { matchScore, confidence: conf, totalElapsed: totalEl, atsScore, semanticScore: semScore, missingSkills: missing, judges: {} }
            setResult(r)
            setStatus('Complete')
            onComplete(r)
          } else if (event === 'early_exit') {
            setStatus(`⚠️ ${ev.detail ?? 'ATS score too low — fix resume formatting before running AI analysis.'}`)
          } else if (event === 'error') {
            setStatus(`Error: ${ev.detail}`)
          }
        } catch { /* skip malformed */ }
      }
    }
  }

  const runAnalysis = useCallback(async () => {
    const resumeIds = [...selectedResumes]
    const jobIds    = [...selectedJobs]
    if (!resumeIds.length || !jobIds.length) { setStatus('Select at least one resume and one job.'); return }

    const pairs = resumeIds.flatMap(rid => jobIds.map(jid => ({ rid, jid })))
    const total  = pairs.length

    setBatchResults([])
    setResult(null)
    setRunning(true)
    setElapsed(0)
    setStatus('Starting…')
    startTimer()
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl

    try {
      for (let i = 0; i < pairs.length; i++) {
        const { rid, jid } = pairs[i]
        const resume = resumes[rid]
        const job    = jobs[jid]
        if (!resume || !job) continue

        resetJudges()
        setResult(null)
        setBatchProgress({ current: i + 1, total, label: `${resume.name} → ${job.title} @ ${job.company}` })
        setStatus(`[${i + 1}/${total}] ${resume.name} × ${job.title}`)

        await streamOne(
          { name: resume.name, text: resume.text },
          { title: job.title, company: job.company, description: job.description },
          ctrl.signal,
          (r) => setBatchResults(prev => [...prev, { resumeName: resume.name, jobTitle: job.title, company: job.company, matchScore: r.matchScore, atsScore: r.atsScore ?? 0, semanticScore: r.semanticScore ?? 0, missingSkills: r.missingSkills ?? [] }]),
        )
      }
      setStatus(total > 1 ? `All ${total} analyses complete` : 'Analysis complete')
    } catch (err) {
      if ((err as Error).name !== 'AbortError') setStatus((err as Error).message)
    } finally {
      stopTimer()
      setRunning(false)
      setBatchProgress(null)
    }
  }, [selectedResumes, selectedJobs, resumes, jobs, model, useCoach])

  useEffect(() => {
    Promise.all([
      api<{ models: { model: string }[]; default_model: string; configured: boolean }>('/openrouter-models').catch(() => null),
      api<{ models: string[] }>('/ollama-models').catch(() => null),
    ]).then(([orRes, olRes]) => {
      const groups: { group: string; models: string[] }[] = []
      if (orRes?.configured && orRes.models.length) {
        groups.push({ group: 'OpenRouter', models: orRes.models.map(m => m.model) })
      }
      if (olRes?.models?.length) {
        groups.push({ group: 'Ollama', models: olRes.models })
      }
      setAvailableModels(groups)
      const defaultModel = orRes?.default_model || olRes?.models?.[0] || ''
      setModel(prev => prev || defaultModel)
    })
  }, [])

  const resumeList = Object.values(resumes)
  const jobList    = Object.values(jobs)

  return (
    <div>
      <PageHeader title="Analyzer" sub="Select a resume and job, then run the 5-judge AI pipeline with live streaming results." />

      <div className="grid gap-4 mb-4" style={{ gridTemplateColumns: '1fr 1fr' }}>
        {/* Resume picker */}
        <Card>
          <CardHeader>
            <CardTitle>Resume</CardTitle>
            <span className="text-[11px] text-[var(--text3)]">{selectedResumes.size} selected</span>
          </CardHeader>
          <div className="flex flex-col gap-1.5 max-h-[220px] overflow-y-auto">
            {resumeList.length === 0 ? <div className="text-[12px] text-[var(--text3)] py-4 text-center">No resumes — add some first</div>
              : resumeList.map(r => {
                const sel = selectedResumes.has(String(r.resume_id))
                const hue = avatarHue(r.resume_id)
                return (
                  <div key={r.resume_id} onClick={() => toggleResume(String(r.resume_id))}
                    className="flex items-center gap-2.5 p-2.5 rounded-md cursor-pointer transition-all duration-100"
                    style={{ border: `1px solid ${sel ? 'var(--accent-border)' : 'var(--border)'}`, background: sel ? 'var(--accent-bg)' : 'var(--surface2)' }}>
                    <div className="w-7 h-7 rounded-md flex items-center justify-center text-[10.5px] font-black font-display flex-shrink-0"
                      style={{ background: `hsla(${hue},55%,50%,.15)`, color: `hsla(${hue},65%,70%,1)` }}>
                      {initials(r.name)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-[12.5px] font-semibold truncate">{r.name}</div>
                      <div className="text-[10.5px] text-[var(--text3)] truncate">{r.title}</div>
                    </div>
                    {sel && <div className="w-4 h-4 rounded-full flex-shrink-0" style={{ background: 'var(--accent)' }} />}
                  </div>
                )
              })}
          </div>
        </Card>

        {/* Job picker */}
        <Card>
          <CardHeader>
            <CardTitle>Job</CardTitle>
            <span className="text-[11px] text-[var(--text3)]">{selectedJobs.size} selected</span>
          </CardHeader>
          <div className="flex flex-col gap-1.5 max-h-[220px] overflow-y-auto">
            {jobList.length === 0 ? <div className="text-[12px] text-[var(--text3)] py-4 text-center">No jobs — add some first</div>
              : jobList.map(j => {
                const sel = selectedJobs.has(String(j.job_id))
                const hue = avatarHue(j.job_id, 83)
                const co  = (j.company || '?')[0].toUpperCase()
                return (
                  <div key={j.job_id} onClick={() => toggleJob(String(j.job_id))}
                    className="flex items-center gap-2.5 p-2.5 rounded-md cursor-pointer transition-all duration-100"
                    style={{ border: `1px solid ${sel ? 'var(--accent-border)' : 'var(--border)'}`, background: sel ? 'var(--accent-bg)' : 'var(--surface2)' }}>
                    <div className="w-7 h-7 rounded-md flex items-center justify-center text-[11px] font-black font-display flex-shrink-0"
                      style={{ background: `hsla(${hue},55%,50%,.15)`, color: `hsla(${hue},65%,70%,1)` }}>
                      {co}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-[12.5px] font-semibold truncate">{j.title}</div>
                      <div className="text-[10.5px] text-[var(--text3)] truncate">{j.company}</div>
                    </div>
                    {sel && <div className="w-4 h-4 rounded-full flex-shrink-0" style={{ background: 'var(--accent)' }} />}
                  </div>
                )
              })}
          </div>
        </Card>
      </div>

      {/* Controls */}
      <Card className="mb-4">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-4">
            <div>
              <label className="block text-[10.5px] font-bold text-[var(--text3)] uppercase tracking-[.5px] mb-1.5">Model</label>
              <div className="relative">
                <select value={model} onChange={e => setModel(e.target.value)}
                  className="appearance-none pr-7 pl-3 py-1.5 text-[12.5px] rounded-md font-body cursor-pointer outline-none transition-all duration-100"
                  style={{ background: 'var(--surface2)', border: '1px solid var(--border2)', color: 'var(--text)', backgroundImage: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'%3E%3Cpath fill='%235C5C72' d='M5 7L1 3h8z'/%3E%3C/svg%3E\")", backgroundRepeat: 'no-repeat', backgroundPosition: 'right 8px center', maxWidth: 320 }}>
                  {availableModels.map(({ group, models }) => (
                    <optgroup key={group} label={group}>
                      {models.map(m => <option key={m} value={m}>{m}</option>)}
                    </optgroup>
                  ))}
                </select>
              </div>
            </div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={useCoach} onChange={e => setUseCoach(e.target.checked)} className="accent-[var(--accent)]" />
              <span className="text-[12.5px] text-[var(--text2)]">Coach Judge</span>
            </label>
          </div>

          <div className="flex items-center gap-3">
            {running && (
              <div className="flex items-center gap-2 text-[12px] text-[var(--text3)] font-mono">
                <div className="w-2 h-2 rounded-full bg-[var(--accent)]" style={{ animation: 'pulse 1.5s ease-in-out infinite' }} />
                {elapsed}s
              </div>
            )}
            <Button onClick={runAnalysis} disabled={running || !selectedResumes.size || !selectedJobs.size} size="md">
              <Play size={13} /> {running ? 'Analyzing…' : `Run ${selectedResumes.size * selectedJobs.size > 1 ? `${selectedResumes.size * selectedJobs.size} Analyses` : 'Analysis'}`}
            </Button>
          </div>
        </div>
        {status && (
          <div className="mt-3 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
            {batchProgress && batchProgress.total > 1 && (
              <div className="mb-2">
                <div className="flex justify-between text-[10.5px] text-[var(--text3)] mb-1">
                  <span>{batchProgress.label}</span>
                  <span>{batchProgress.current}/{batchProgress.total}</span>
                </div>
                <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--surface3)' }}>
                  <div className="h-full rounded-full transition-all duration-500" style={{ width: `${(batchProgress.current / batchProgress.total) * 100}%`, background: 'var(--accent)' }} />
                </div>
              </div>
            )}
            <div className="text-[11.5px] text-[var(--text3)]">{status}</div>
          </div>
        )}
      </Card>

      {/* Selection summary */}
      {!running && !result && (selectedResumes.size > 0 || selectedJobs.size > 0) && (
        <div className="flex items-center gap-2 mb-4 text-[12px] text-[var(--text2)] flex-wrap">
          <span className="px-2.5 py-1 rounded-md" style={{ background: 'var(--surface2)', border: '1px solid var(--border)' }}>📄 {selectedResumes.size} resume{selectedResumes.size !== 1 ? 's' : ''}</span>
          <ChevronDown size={14} style={{ transform: 'rotate(-90deg)', color: 'var(--text3)' }} />
          <span className="px-2.5 py-1 rounded-md" style={{ background: 'var(--surface2)', border: '1px solid var(--border)' }}>💼 {selectedJobs.size} job{selectedJobs.size !== 1 ? 's' : ''}</span>
          {selectedResumes.size > 0 && selectedJobs.size > 0 && (
            <span className="text-[11px] text-[var(--text3)]">= {selectedResumes.size * selectedJobs.size} run{selectedResumes.size * selectedJobs.size !== 1 ? 's' : ''}</span>
          )}
        </div>
      )}

      {/* Pipeline + Result */}
      {(running || result) && (
        <div className="grid gap-4" style={{ gridTemplateColumns: '1fr 320px' }}>
          {/* Judge pipeline */}
          <Card>
            <CardHeader>
              <CardTitle>Judge Pipeline</CardTitle>
              {running && <div className="text-[11px] text-[var(--text3)] font-mono">{elapsed}s</div>}
            </CardHeader>
            <div className="flex flex-col gap-2.5">
              {JUDGES.map(judge => (
                <JudgeCard key={judge.id} judge={judge} result={judgeStates[judge.id] ?? { state: 'pending', score: 0, elapsed: 0 }} />
              ))}
            </div>
          </Card>

          {/* Score panel */}
          <div className="flex flex-col gap-4">
            <Card className="flex flex-col items-center py-6">
              <div className="text-[11px] font-bold text-[var(--text3)] uppercase tracking-[.8px] mb-4">Match Score</div>
              <ScoreRing score={result?.matchScore ?? 0} size={140} strokeWidth={10} animate={!!result} />
              {result && (
                <div className="mt-4 w-full flex flex-col gap-2">
                  {[
                    { label: 'Confidence', value: result.confidence, isScore: true },
                    { label: 'ATS Score',  value: result.atsScore ?? 0, isScore: true },
                    { label: 'Semantic',   value: result.semanticScore ?? 0, isScore: true },
                    { label: 'Time',       value: `${result.totalElapsed.toFixed(1)}s`, isScore: false },
                  ].map(({ label, value, isScore }) => (
                    <div key={label} className="flex items-center justify-between px-1">
                      <span className="text-[11px] text-[var(--text3)]">{label}</span>
                      {isScore ? <ScoreBadge score={value as number} /> : <span className="text-[11.5px] font-mono text-[var(--text2)]">{value}</span>}
                    </div>
                  ))}
                </div>
              )}
            </Card>

            {result?.missingSkills && result.missingSkills.length > 0 && (
              <Card>
                <CardTitle className="mb-3">Missing Skills</CardTitle>
                <div className="flex flex-wrap gap-1.5">
                  {result.missingSkills.map(s => (
                    <span key={s} className="text-[11px] font-semibold px-2 py-0.5 rounded" style={{ background: 'var(--red-bg)', color: 'var(--red)', border: '1px solid var(--red-border)' }}>{s}</span>
                  ))}
                </div>
              </Card>
            )}
          </div>
        </div>
      )}

      {/* Batch results table */}
      {batchResults.length > 1 && (
        <Card className="mt-4">
          <CardHeader>
            <CardTitle>Batch Results</CardTitle>
            <span className="text-[11.5px] text-[var(--text3)]">{batchResults.length} / {selectedResumes.size * selectedJobs.size} complete</span>
          </CardHeader>
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]">
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['Candidate', 'Job', 'Company', 'Match', 'ATS', 'Semantic'].map(h => (
                    <th key={h} className="text-left pb-2 pr-4 text-[10.5px] font-bold text-[var(--text3)] uppercase tracking-wider">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {batchResults.map((r, i) => {
                  const sc = (s: number) => s >= 70 ? 'var(--green)' : s >= 40 ? 'var(--amber)' : 'var(--red)'
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td className="py-2.5 pr-4 font-semibold">{r.resumeName}</td>
                      <td className="py-2.5 pr-4 text-[var(--text2)] max-w-[180px] truncate">{r.jobTitle}</td>
                      <td className="py-2.5 pr-4 text-[var(--text3)]">{r.company}</td>
                      <td className="py-2.5 pr-4 font-black font-display text-[13px]" style={{ color: sc(r.matchScore) }}>{r.matchScore}%</td>
                      <td className="py-2.5 pr-4 font-mono" style={{ color: sc(r.atsScore) }}>{r.atsScore}%</td>
                      <td className="py-2.5 font-mono" style={{ color: sc(r.semanticScore) }}>{r.semanticScore}%</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <style>{`
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }
        @keyframes spin  { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}
