import { useState } from 'react'
import { useStore } from '@/store'
import { api } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { initials, avatarHue } from '@/lib/utils'
import { Search, Zap, Brain } from 'lucide-react'

function PageHeader({ title, sub }: { title: string; sub: string }) {
  return (
    <div className="relative pb-[18px] mb-6" style={{ borderBottom: '1px solid var(--border)' }}>
      <div className="absolute bottom-[-1px] left-0 w-10 h-0.5 rounded-sm bg-[var(--accent)]" />
      <h1 className="font-display text-[20px] font-black tracking-[-0.5px] mb-1">{title}</h1>
      <p className="text-[13px] text-[var(--text2)]">{sub}</p>
    </div>
  )
}

const inputBase: React.CSSProperties = { background: 'var(--surface2)', border: '1px solid var(--border2)', color: 'var(--text)' }
const onFocus = (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>) => {
  e.target.style.borderColor = 'var(--accent)'; e.target.style.boxShadow = '0 0 0 3px var(--accent-glow)'
}
const onBlur = (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>) => {
  e.target.style.borderColor = 'var(--border2)'; e.target.style.boxShadow = 'none'
}

function verdictColor(v: string) {
  if (v === 'PASS') return 'var(--green)'
  if (v === 'FAIL') return 'var(--red)'
  return 'var(--amber)'
}

function ScoreDim({ label, score, verdict, note, improvement }: { label: string; score: number; verdict: string; note: string; improvement: string }) {
  const c = verdictColor(verdict)
  return (
    <div className="p-3 rounded-lg mb-2" style={{ background: 'var(--surface2)', border: `1px solid ${c}33` }}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-[12px] font-bold">{label}</span>
        <div className="flex items-center gap-2">
          <span className="text-[12.5px] font-black font-mono" style={{ color: c }}>{score.toFixed(1)}</span>
          <span className="text-[9.5px] font-bold px-2 py-0.5 rounded-full" style={{ background: `${c}18`, color: c, border: `1px solid ${c}44` }}>{verdict}</span>
        </div>
      </div>
      <p className="text-[11px] text-[var(--text2)] mb-1">{note}</p>
      {improvement && <p className="text-[10.5px] text-[var(--text3)] italic">{improvement}</p>}
    </div>
  )
}

export function AtsChecker() {
  const { resumes } = useStore()
  const [resumeText, setResumeText] = useState('')
  const [jd, setJd] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [filter, setFilter] = useState('')
  const [quickResult, setQuickResult] = useState<Record<string, unknown> | null>(null)
  const [deepResult, setDeepResult] = useState<Record<string, unknown> | null>(null)
  const [quickRunning, setQuickRunning] = useState(false)
  const [deepRunning, setDeepRunning] = useState(false)
  const [error, setError] = useState('')

  const items = Object.values(resumes).filter(r =>
    !filter || r.name?.toLowerCase().includes(filter.toLowerCase())
  )

  const selectResume = (id: string) => {
    const r = resumes[id]
    if (!r) return
    setSelectedId(id)
    setResumeText(r.text ?? '')
    setQuickResult(null); setDeepResult(null); setError('')
  }

  const runQuick = async () => {
    if (!resumeText.trim()) { setError('Resume text required.'); return }
    setQuickRunning(true); setError(''); setQuickResult(null)
    try {
      const res = await api<Record<string, unknown>>('/ats-check', { method: 'POST', body: JSON.stringify({ resume_text: resumeText, job_description: jd }) })
      setQuickResult(res)
    } catch (e) { setError(e instanceof Error ? e.message : 'Request failed') }
    finally { setQuickRunning(false) }
  }

  const runDeep = async () => {
    if (!resumeText.trim()) { setError('Resume text required.'); return }
    setDeepRunning(true); setError(''); setDeepResult(null)
    try {
      const res = await api<Record<string, unknown>>('/ats-deep-audit', { method: 'POST', body: JSON.stringify({ resume_text: resumeText, job_description: jd }) })
      setDeepResult(res)
    } catch (e) { setError(e instanceof Error ? e.message : 'Request failed') }
    finally { setDeepRunning(false) }
  }

  const scorecard = deepResult?.scorecard as Record<string, { score: number; verdict: string; note: string; improvement: string }> | undefined
  const disqualifiers = deepResult?.disqualifiers as { rank: number; title: string; explanation: string }[] | undefined
  const rewrites = deepResult?.rewrites as { original: string; why_fails: string; refactored: string }[] | undefined
  const overallVerdict = scorecard?.overall_verdict as string | undefined

  const quickParsed = quickResult as {
    overall_score?: number; grade?: string; section_quality?: Record<string, string>;
    keyword_density?: { present?: string[]; missing?: string[] }
    formatting?: { issues?: string[] }; recommendations?: string[]
  } | null

  return (
    <div>
      <PageHeader title="ATS Checker" sub="Detect parsing failures, keyword gaps, and structural issues before submitting." />
      <div className="grid gap-4" style={{ gridTemplateColumns: 'minmax(240px,.8fr) minmax(400px,1.2fr)' }}>
        {/* Left: picker + inputs */}
        <div className="flex flex-col gap-4">
          <Card>
            <CardHeader><CardTitle>Select Resume</CardTitle></CardHeader>
            <div className="flex items-center gap-2 px-2.5 rounded-md mb-2.5" style={{ background: 'var(--surface2)', border: '1px solid var(--border)' }}>
              <Search size={12} color="var(--text3)" />
              <input value={filter} onChange={e => setFilter(e.target.value)} placeholder="Filter…"
                className="border-0 bg-transparent outline-none text-[12px] py-2 flex-1 text-[var(--text)] font-body" />
            </div>
            <div className="flex flex-col gap-1.5 max-h-[220px] overflow-y-auto">
              {items.length === 0 ? (
                <div className="text-center py-6 text-[var(--text3)] text-[12px]">No resumes</div>
              ) : items.map(r => {
                const sel = selectedId === String(r.resume_id)
                const hue = avatarHue(r.resume_id)
                return (
                  <div key={r.resume_id} onClick={() => selectResume(String(r.resume_id))}
                    className="flex items-center gap-2 p-2 rounded-md cursor-pointer transition-all duration-100"
                    style={{ border: `1px solid ${sel ? 'var(--accent-border)' : 'var(--border)'}`, background: sel ? 'var(--accent-bg)' : 'var(--surface2)' }}>
                    <div className="w-6 h-6 rounded flex items-center justify-center text-[9.5px] font-black flex-shrink-0"
                      style={{ background: `hsla(${hue},55%,50%,.15)`, color: `hsla(${hue},65%,70%,1)` }}>
                      {initials(r.name)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-[12px] font-semibold truncate">{r.name}</div>
                      <div className="text-[10.5px] text-[var(--text3)]">{r.title}</div>
                    </div>
                  </div>
                )
              })}
            </div>
          </Card>

          <Card>
            <CardHeader><CardTitle>Resume Text</CardTitle></CardHeader>
            <textarea value={resumeText} onChange={e => setResumeText(e.target.value)} rows={7}
              placeholder="Paste resume text or select from library…"
              className="w-full px-3 py-2 text-[11.5px] rounded-md outline-none font-mono resize-y"
              style={inputBase} onFocus={onFocus} onBlur={onBlur} />
          </Card>

          <Card>
            <CardHeader><CardTitle>Job Description <span className="text-[var(--text3)] font-normal">(optional)</span></CardTitle></CardHeader>
            <textarea value={jd} onChange={e => setJd(e.target.value)} rows={4}
              placeholder="Paste JD for role-targeting analysis…"
              className="w-full px-3 py-2 text-[11.5px] rounded-md outline-none font-body resize-y"
              style={inputBase} onFocus={onFocus} onBlur={onBlur} />
            <div className="flex gap-2 mt-3">
              <Button variant="secondary" size="sm" className="flex-1" onClick={runQuick} disabled={quickRunning}>
                <Zap size={12} />{quickRunning ? 'Running…' : 'Quick Check'}
              </Button>
              <Button size="sm" className="flex-1" onClick={runDeep} disabled={deepRunning}>
                <Brain size={12} />{deepRunning ? 'Auditing…' : 'Deep Audit'}
              </Button>
            </div>
            {error && <div className="text-[11.5px] mt-2" style={{ color: 'var(--red)' }}>{error}</div>}
          </Card>
        </div>

        {/* Right: results */}
        <div className="flex flex-col gap-4">
          {!quickResult && !deepResult && !deepRunning && !quickRunning && (
            <Card>
              <div className="text-center py-16 text-[var(--text3)]">
                <div className="text-[32px] mb-3 opacity-30">🔍</div>
                <div className="text-[13px] font-semibold">No results yet</div>
                <div className="text-[12px] mt-1">Select a resume and run Quick Check or Deep Audit.</div>
              </div>
            </Card>
          )}

          {(quickRunning || deepRunning) && (
            <Card>
              <div className="text-center py-16 text-[var(--text3)]">
                <div className="text-[13px] font-semibold">{deepRunning ? 'Running deep audit via LLM…' : 'Checking ATS rules…'}</div>
                <div className="text-[12px] mt-1 text-[var(--text3)]">{deepRunning ? 'This may take 20–60 seconds.' : 'A few seconds…'}</div>
              </div>
            </Card>
          )}

          {quickParsed && (
            <Card>
              <CardHeader>
                <CardTitle>Quick ATS Check</CardTitle>
                {quickParsed.grade && (
                  <span className="text-[18px] font-black font-display" style={{ color: verdictColor(quickParsed.grade === 'A' || quickParsed.grade === 'B' ? 'PASS' : quickParsed.grade === 'C' ? 'REVIEW' : 'FAIL') }}>
                    {quickParsed.grade}
                  </span>
                )}
              </CardHeader>
              {quickParsed.overall_score !== undefined && (
                <div className="mb-3 text-[28px] font-black font-display" style={{ color: verdictColor(quickParsed.overall_score >= 70 ? 'PASS' : quickParsed.overall_score >= 40 ? 'REVIEW' : 'FAIL') }}>
                  {typeof quickParsed.overall_score === 'number' ? quickParsed.overall_score.toFixed(0) : quickParsed.overall_score}%
                </div>
              )}
              {quickParsed.keyword_density?.present && quickParsed.keyword_density.present.length > 0 && (
                <div className="mb-3">
                  <div className="text-[11px] font-bold text-[var(--text3)] uppercase tracking-wider mb-1.5">Keywords Found</div>
                  <div className="flex flex-wrap gap-1">
                    {quickParsed.keyword_density.present.slice(0, 12).map(k => (
                      <span key={k} className="text-[10.5px] px-2 py-0.5 rounded" style={{ background: 'var(--green-bg)', color: 'var(--green)', border: '1px solid var(--green-border)' }}>{k}</span>
                    ))}
                  </div>
                </div>
              )}
              {quickParsed.keyword_density?.missing && quickParsed.keyword_density.missing.length > 0 && (
                <div className="mb-3">
                  <div className="text-[11px] font-bold text-[var(--text3)] uppercase tracking-wider mb-1.5">Keywords Missing</div>
                  <div className="flex flex-wrap gap-1">
                    {quickParsed.keyword_density.missing.slice(0, 12).map(k => (
                      <span key={k} className="text-[10.5px] px-2 py-0.5 rounded" style={{ background: 'var(--red-bg)', color: 'var(--red)', border: '1px solid var(--red-border)' }}>{k}</span>
                    ))}
                  </div>
                </div>
              )}
              {quickParsed.recommendations && quickParsed.recommendations.length > 0 && (
                <div>
                  <div className="text-[11px] font-bold text-[var(--text3)] uppercase tracking-wider mb-1.5">Recommendations</div>
                  <ul className="flex flex-col gap-1">
                    {quickParsed.recommendations.map((r, i) => (
                      <li key={i} className="text-[11.5px] text-[var(--text2)] flex gap-2"><span style={{ color: 'var(--accent)' }}>›</span>{r}</li>
                    ))}
                  </ul>
                </div>
              )}
            </Card>
          )}

          {deepResult && scorecard && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>ATS Deep Audit</CardTitle>
                  {overallVerdict && (
                    <span className="text-[11px] font-black px-3 py-1 rounded-full" style={{ background: `${verdictColor(overallVerdict)}18`, color: verdictColor(overallVerdict), border: `1px solid ${verdictColor(overallVerdict)}44` }}>
                      {overallVerdict}
                    </span>
                  )}
                </CardHeader>
                {(['ats_parsability', 'metric_caliber', 'role_targeting', 'semantic_red_flags'] as const).map(key => {
                  const dim = scorecard[key]
                  if (!dim) return null
                  const labels: Record<string, string> = { ats_parsability: 'ATS Parsability', metric_caliber: 'Metric Caliber', role_targeting: 'Role Targeting', semantic_red_flags: 'Semantic Red Flags' }
                  return <ScoreDim key={key} label={labels[key]} score={dim.score} verdict={dim.verdict} note={dim.note} improvement={dim.improvement} />
                })}
              </Card>

              {disqualifiers && disqualifiers.length > 0 && (
                <Card>
                  <CardHeader><CardTitle>Disqualifiers</CardTitle></CardHeader>
                  <div className="flex flex-col gap-2">
                    {disqualifiers.map(d => (
                      <div key={d.rank} className="p-3 rounded-lg" style={{ background: 'var(--red-bg)', border: '1px solid var(--red-border)' }}>
                        <div className="text-[12px] font-bold mb-0.5" style={{ color: 'var(--red)' }}>#{d.rank} {d.title}</div>
                        <div className="text-[11.5px] text-[var(--text2)]">{d.explanation}</div>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {rewrites && rewrites.length > 0 && (
                <Card>
                  <CardHeader><CardTitle>Suggested Rewrites</CardTitle></CardHeader>
                  <div className="flex flex-col gap-3">
                    {rewrites.map((rw, i) => (
                      <div key={i} className="p-3 rounded-lg" style={{ background: 'var(--surface2)', border: '1px solid var(--border)' }}>
                        <div className="text-[10.5px] text-[var(--text3)] font-bold uppercase tracking-wider mb-1">Original</div>
                        <div className="text-[11.5px] text-[var(--text2)] mb-1.5 font-mono bg-[var(--surface3)] px-2 py-1 rounded">{rw.original}</div>
                        <div className="text-[10.5px] font-bold uppercase tracking-wider mb-1" style={{ color: 'var(--red)' }}>Why it fails: <span className="font-normal text-[var(--text2)] normal-case tracking-normal">{rw.why_fails}</span></div>
                        <div className="text-[10.5px] font-bold uppercase tracking-wider mb-1" style={{ color: 'var(--green)' }}>Refactored</div>
                        <div className="text-[11.5px] font-mono px-2 py-1 rounded" style={{ background: 'var(--green-bg)', color: 'var(--green)', border: '1px solid var(--green-border)' }}>{rw.refactored}</div>
                      </div>
                    ))}
                  </div>
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
