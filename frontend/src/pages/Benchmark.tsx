import { useEffect, useRef, useState } from 'react'
import { useStore } from '@/store'
import { api } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Play, RefreshCw, Trophy } from 'lucide-react'
import { initials, avatarHue } from '@/lib/utils'

function PageHeader({ title, sub }: { title: string; sub: string }) {
  return (
    <div className="relative pb-[18px] mb-6" style={{ borderBottom: '1px solid var(--border)' }}>
      <div className="absolute bottom-[-1px] left-0 w-10 h-0.5 rounded-sm bg-[var(--accent)]" />
      <h1 className="font-display text-[20px] font-black tracking-[-0.5px] mb-1">{title}</h1>
      <p className="text-[13px] text-[var(--text2)]">{sub}</p>
    </div>
  )
}

const JUDGE_NAMES = ['skills', 'experience', 'seniority', 'relevancy', 'impact'] as const

interface JudgeStats {
  pass_rate: number
  avg_confidence: number
  avg_latency_s: number
  first_attempt_rate: number
  truncation_rate: number
  avg_score: number
  score_variance: number
}

interface ModelResult {
  model: string
  composite: number
  avg_judge_score: number
  avg_pass_rate: number
  avg_confidence: number
  avg_latency_s: number
  avg_speed_score: number
  avg_first_attempt_rate: number
  per_judge: Record<string, JudgeStats>
  audit: { parse_success: boolean; latency_s: number; error?: string } | null
}

interface BenchData {
  timestamp: string
  judge_resume: string
  audit_resume: string
  jobs_tested: string[]
  models_tested: number
  results: ModelResult[]
  recommended: { judge: string; audit: string }
}

function compositeColor(v: number) {
  if (v >= 75) return 'var(--green)'
  if (v >= 50) return 'var(--amber)'
  return 'var(--red)'
}

function Bar({ pct, color, height = 6 }: { pct: number; color: string; height?: number }) {
  return (
    <div className="rounded-full overflow-hidden flex-1" style={{ background: 'var(--surface3)', height }}>
      <div className="h-full rounded-full transition-all duration-700" style={{ width: `${Math.min(pct, 100)}%`, background: color }} />
    </div>
  )
}

function ModelRow({ result, rank, expanded, onToggle }: {
  result: ModelResult; rank: number; expanded: boolean; onToggle: () => void
}) {
  const c = compositeColor(result.composite)
  const isWinner = rank === 1
  return (
    <div className="rounded-[var(--radius)] overflow-hidden mb-2" style={{ border: `1px solid ${isWinner ? 'var(--accent-border)' : 'var(--border)'}` }}>
      <div className="flex items-center gap-3 p-3 cursor-pointer transition-colors hover:bg-[var(--surface2)]"
        style={{ background: isWinner ? 'var(--accent-bg)' : 'var(--surface)' }}
        onClick={onToggle}>
        <div className="w-7 h-7 rounded-lg flex items-center justify-center text-[11px] font-black font-display flex-shrink-0"
          style={{ background: isWinner ? 'var(--accent-bg)' : 'var(--surface3)', color: isWinner ? 'var(--accent-light)' : 'var(--text3)', border: isWinner ? '1px solid var(--accent-border)' : 'none' }}>
          {isWinner ? <Trophy size={13} /> : rank}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-[12.5px] font-bold font-mono truncate">{result.model}</span>
            {isWinner && <span className="text-[9px] font-black px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-bg)', color: 'var(--accent-light)', border: '1px solid var(--accent-border)' }}>BEST</span>}
          </div>
          <div className="flex items-center gap-2">
            <Bar pct={result.composite} color={c} />
            <span className="text-[12px] font-black font-display flex-shrink-0" style={{ color: c }}>{result.composite.toFixed(1)}</span>
          </div>
        </div>
        <div className="grid grid-cols-4 gap-4 flex-shrink-0 text-right">
          {[['Pass', result.avg_pass_rate], ['Conf', result.avg_confidence], ['Latency', null], ['Speed', result.avg_speed_score]].map(([label, val]) => (
            <div key={label as string}>
              <div className="text-[10px] text-[var(--text3)] font-bold uppercase">{label}</div>
              <div className="text-[12px] font-bold font-mono">
                {label === 'Latency' ? `${result.avg_latency_s.toFixed(1)}s` : `${(val as number).toFixed(1)}%`}
              </div>
            </div>
          ))}
        </div>
      </div>

      {expanded && (
        <div className="px-4 pb-4 pt-2" style={{ borderTop: '1px solid var(--border)', background: 'var(--surface2)' }}>
          <div className="text-[10.5px] font-bold text-[var(--text3)] uppercase tracking-wider mb-2.5">Per-Judge Breakdown</div>
          <div className="grid grid-cols-5 gap-2 mb-3">
            {JUDGE_NAMES.map(j => {
              const s = result.per_judge[j]
              if (!s) return <div key={j} className="text-center text-[var(--text3)] text-[10px]">—</div>
              const jc = s.pass_rate >= 80 ? 'var(--green)' : s.pass_rate >= 50 ? 'var(--amber)' : 'var(--red)'
              return (
                <div key={j} className="p-2 rounded-md text-center" style={{ background: 'var(--surface3)', border: '1px solid var(--border)' }}>
                  <div className="text-[10px] font-bold text-[var(--text3)] uppercase mb-1">{j}</div>
                  <div className="text-[15px] font-black font-display mb-0.5" style={{ color: jc }}>{s.pass_rate.toFixed(0)}%</div>
                  <div className="text-[9.5px] text-[var(--text3)]">pass</div>
                  <div className="text-[10.5px] font-mono mt-1" style={{ color: 'var(--text2)' }}>{s.avg_latency_s.toFixed(1)}s</div>
                  <div className="text-[9.5px] text-[var(--text3)]">avg latency</div>
                </div>
              )
            })}
          </div>
          {result.audit && (
            <div className="flex items-center gap-3 p-2 rounded-md" style={{ background: 'var(--surface3)', border: '1px solid var(--border)' }}>
              <span className="text-[11px] font-bold text-[var(--text3)] uppercase">ATS Audit</span>
              <span className="w-2 h-2 rounded-full" style={{ background: result.audit.parse_success ? 'var(--green)' : 'var(--red)' }} />
              <span className="text-[11px]" style={{ color: result.audit.parse_success ? 'var(--green)' : 'var(--red)' }}>
                {result.audit.parse_success ? 'Parsed' : 'Failed'}
              </span>
              <span className="text-[11px] font-mono text-[var(--text3)]">{result.audit.latency_s.toFixed(1)}s</span>
              {result.audit.error && <span className="text-[10px] text-[var(--red)] truncate">{result.audit.error}</span>}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function Benchmark() {
  const { resumes, jobs } = useStore()
  const [data, setData] = useState<BenchData | null>(null)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState<string | null>(null)
  const [progress, setProgress] = useState<{ pct: number; completed: number; total: number; current_model: string; current_judge: string; current_job: string; model_index: number; models_total: number } | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // run config
  const [judgeResumeId, setJudgeResumeId] = useState('')
  const [auditResumeId, setAuditResumeId] = useState('')
  const [selectedJobs, setSelectedJobs] = useState<Set<string>>(new Set())

  const resumeList = Object.values(resumes)
  const jobList = Object.values(jobs)

  const toggleJob = (id: string) => {
    setSelectedJobs(prev => {
      const s = new Set(prev)
      if (s.has(id)) s.delete(id)
      else if (s.size < 3) s.add(id)
      return s
    })
  }

  const load = async () => {
    setLoading(true)
    try { setData(await api<BenchData>('/benchmark-ollama-full-results')) }
    catch { /* no results yet */ }
    finally { setLoading(false) }
  }

  useEffect(() => { void load() }, [])

  const startPolling = () => {
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const p = await api<typeof progress>('/benchmark-ollama-full-progress')
        setProgress(p)
      } catch { /* ignore */ }
    }, 4000)
  }

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }

  useEffect(() => () => stopPolling(), [])

  const run = async () => {
    if (!judgeResumeId || !auditResumeId || selectedJobs.size === 0) {
      setError('Select judge resume, audit resume, and at least 1 job (max 3).'); return
    }
    setRunning(true); setError(''); setProgress(null)
    startPolling()
    try {
      const result = await api<BenchData>('/benchmark-ollama-full', {
        method: 'POST',
        body: JSON.stringify({ judge_resume_id: judgeResumeId, audit_resume_id: auditResumeId, job_ids: [...selectedJobs] }),
      })
      setData(result)
    } catch (e) { setError(e instanceof Error ? e.message : 'Benchmark failed') }
    finally { setRunning(false); stopPolling(); setProgress(null) }
  }

  const fmt = (ts: string) => {
    try { return new Date(ts).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) }
    catch { return ts }
  }

  return (
    <div>
      <PageHeader title="Benchmark" sub="Compare all local Ollama models across 5 judges × multiple JDs to find the best performer." />

      <div className="grid gap-4 mb-5" style={{ gridTemplateColumns: '1fr 1fr 1fr' }}>
        {/* Judge resume */}
        <Card>
          <CardHeader><CardTitle>Judge Resume</CardTitle><span className="text-[10.5px] text-[var(--text3)]">scored against JDs</span></CardHeader>
          <div className="flex flex-col gap-1 max-h-[180px] overflow-y-auto">
            {resumeList.map(r => {
              const sel = judgeResumeId === String(r.resume_id)
              const hue = avatarHue(r.resume_id)
              return (
                <div key={r.resume_id} onClick={() => setJudgeResumeId(String(r.resume_id))}
                  className="flex items-center gap-2 p-2 rounded-md cursor-pointer transition-all"
                  style={{ border: `1px solid ${sel ? 'var(--accent-border)' : 'var(--border)'}`, background: sel ? 'var(--accent-bg)' : 'var(--surface2)' }}>
                  <div className="w-6 h-6 rounded text-[9.5px] font-black flex items-center justify-center flex-shrink-0"
                    style={{ background: `hsla(${hue},55%,50%,.15)`, color: `hsla(${hue},65%,70%,1)` }}>
                    {initials(r.name)}
                  </div>
                  <span className="text-[11.5px] font-semibold truncate">{r.name}</span>
                </div>
              )
            })}
          </div>
        </Card>

        {/* Audit resume */}
        <Card>
          <CardHeader><CardTitle>Audit Resume</CardTitle><span className="text-[10.5px] text-[var(--text3)]">for ATS deep audit</span></CardHeader>
          <div className="flex flex-col gap-1 max-h-[180px] overflow-y-auto">
            {resumeList.map(r => {
              const sel = auditResumeId === String(r.resume_id)
              const hue = avatarHue(r.resume_id)
              return (
                <div key={r.resume_id} onClick={() => setAuditResumeId(String(r.resume_id))}
                  className="flex items-center gap-2 p-2 rounded-md cursor-pointer transition-all"
                  style={{ border: `1px solid ${sel ? 'var(--green-border)' : 'var(--border)'}`, background: sel ? 'var(--green-bg)' : 'var(--surface2)' }}>
                  <div className="w-6 h-6 rounded text-[9.5px] font-black flex items-center justify-center flex-shrink-0"
                    style={{ background: `hsla(${hue},55%,50%,.15)`, color: `hsla(${hue},65%,70%,1)` }}>
                    {initials(r.name)}
                  </div>
                  <span className="text-[11.5px] font-semibold truncate">{r.name}</span>
                </div>
              )
            })}
          </div>
        </Card>

        {/* Jobs */}
        <Card>
          <CardHeader><CardTitle>Test Jobs</CardTitle><span className="text-[10.5px] text-[var(--text3)]">{selectedJobs.size}/3 selected</span></CardHeader>
          <div className="flex flex-col gap-1 max-h-[180px] overflow-y-auto">
            {jobList.map(j => {
              const sel = selectedJobs.has(String(j.job_id))
              return (
                <div key={j.job_id} onClick={() => toggleJob(String(j.job_id))}
                  className="flex items-center gap-2 p-2 rounded-md cursor-pointer transition-all"
                  style={{ border: `1px solid ${sel ? 'var(--amber-border)' : 'var(--border)'}`, background: sel ? 'var(--amber-bg)' : 'var(--surface2)', opacity: !sel && selectedJobs.size >= 3 ? 0.4 : 1 }}>
                  <span className="text-[11.5px] font-semibold truncate">{j.title}</span>
                  <span className="text-[10px] text-[var(--text3)] flex-shrink-0 ml-auto">{j.company}</span>
                </div>
              )
            })}
          </div>
        </Card>
      </div>

      <div className="flex items-center gap-3 mb-5">
        <Button onClick={run} disabled={running}>
          <Play size={13} />{running ? 'Benchmarking all models…' : 'Run Full Benchmark'}
        </Button>
        <Button variant="secondary" onClick={load} disabled={loading}>
          <RefreshCw size={13} /> Refresh
        </Button>
        {error && <span className="text-[11.5px]" style={{ color: 'var(--red)' }}>{error}</span>}
      </div>

      {running && (
        <div className="mb-5 p-4 rounded-[var(--radius)]" style={{ background: 'var(--surface)', border: '1px solid var(--accent-border)' }}>
          <div className="flex items-center justify-between mb-2">
            <div>
              <span className="text-[12.5px] font-bold">Benchmarking in progress</span>
              {progress && (
                <span className="text-[11px] text-[var(--text3)] ml-2">
                  Model {progress.model_index}/{progress.models_total} — {progress.current_model}
                </span>
              )}
            </div>
            <span className="text-[13px] font-black font-mono" style={{ color: 'var(--accent-light)' }}>
              {progress ? `${progress.pct.toFixed(0)}%` : '…'}
            </span>
          </div>
          <div className="h-2 rounded-full overflow-hidden mb-2" style={{ background: 'var(--surface3)' }}>
            {progress ? (
              <div className="h-full rounded-full transition-all duration-700"
                style={{ width: `${progress.pct}%`, background: 'linear-gradient(90deg, var(--accent) 0%, var(--accent-light) 100%)' }} />
            ) : (
              <div className="h-full rounded-full animate-pulse" style={{ width: '15%', background: 'var(--accent)', animation: 'pulse 1.5s ease-in-out infinite' }} />
            )}
          </div>
          {progress && (
            <div className="flex items-center gap-4 text-[10.5px] text-[var(--text3)]">
              <span>{progress.completed}/{progress.total} calls</span>
              <span className="font-mono">judge: <span style={{ color: 'var(--accent-light)' }}>{progress.current_judge}</span></span>
              <span className="truncate max-w-[300px]">job: {progress.current_job}</span>
            </div>
          )}
        </div>
      )}

      {loading ? (
        <Card><div className="text-center py-10 text-[var(--text3)] text-[13px]">Loading…</div></Card>
      ) : !data || !('results' in data) ? (
        <Card>
          <div className="text-center py-14 text-[var(--text3)]">
            <div className="text-[32px] mb-3 opacity-30">📈</div>
            <div className="text-[13px] font-semibold">No benchmark data yet</div>
            <div className="text-[12px] mt-1">Select resumes and jobs above, then click Run Full Benchmark.</div>
          </div>
        </Card>
      ) : (
        <div>
          <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
            <div className="text-[12.5px] font-semibold text-[var(--text2)]">
              {data.results.length} models · ran {fmt(data.timestamp)} · tested on {data.jobs_tested.length} JD{data.jobs_tested.length !== 1 ? 's' : ''}
            </div>
            <div className="flex gap-2 flex-wrap">
              {data.jobs_tested.map(j => (
                <span key={j} className="text-[10px] px-2 py-0.5 rounded font-mono" style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text3)' }}>{j}</span>
              ))}
            </div>
          </div>
          {data.results.map((r, i) => (
            <ModelRow key={r.model} result={r} rank={i + 1}
              expanded={expanded === r.model}
              onToggle={() => setExpanded(expanded === r.model ? null : r.model)} />
          ))}
        </div>
      )}
    </div>
  )
}
