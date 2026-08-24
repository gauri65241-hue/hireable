import { useState } from 'react'
import { useStore } from '@/store'
import { api } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { initials, avatarHue } from '@/lib/utils'
import { Search, TrendingUp } from 'lucide-react'

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
const onFocus = (e: React.FocusEvent<HTMLTextAreaElement>) => {
  e.target.style.borderColor = 'var(--accent)'; e.target.style.boxShadow = '0 0 0 3px var(--accent-glow)'
}
const onBlur = (e: React.FocusEvent<HTMLTextAreaElement>) => {
  e.target.style.borderColor = 'var(--border2)'; e.target.style.boxShadow = 'none'
}

function demandColor(d: string) {
  const l = d?.toLowerCase() ?? ''
  if (l === 'high' || l === 'very high') return 'var(--green)'
  if (l === 'medium') return 'var(--amber)'
  return 'var(--red)'
}

interface PathMatch {
  id: string
  title: string
  category: string
  description: string
  income_range: string
  time_to_first_income: string
  market_demand: string
  match_pct: number
  matched_skills: string[]
  missing_required: string[]
  starter_actions: string[]
}

function PathCard({ path, rank }: { path: PathMatch; rank: number }) {
  const [expanded, setExpanded] = useState(false)
  const pct = path.match_pct
  const pctColor = pct >= 70 ? 'var(--green)' : pct >= 40 ? 'var(--amber)' : 'var(--red)'

  return (
    <div className="p-4 rounded-[var(--radius)] transition-all duration-100 hover:border-[var(--border2)] cursor-pointer"
      style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
      onClick={() => setExpanded(e => !e)}>
      <div className="flex items-start gap-3">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center text-[11px] font-black font-display flex-shrink-0"
          style={{ background: 'var(--accent-bg)', color: 'var(--accent-light)', border: '1px solid var(--accent-border)' }}>
          {rank}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2 mb-0.5">
            <span className="text-[13px] font-bold">{path.title}</span>
            <span className="text-[18px] font-black font-display flex-shrink-0" style={{ color: pctColor }}>{pct}%</span>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[10px] px-2 py-0.5 rounded-full font-semibold" style={{ background: 'var(--surface3)', color: 'var(--text3)' }}>{path.category}</span>
            <span className="text-[10px] font-bold" style={{ color: demandColor(path.market_demand) }}>{path.market_demand} demand</span>
            <span className="text-[10px] text-[var(--text3)]">{path.income_range}</span>
          </div>

          <div className="mt-2 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--surface3)' }}>
            <div className="h-full rounded-full" style={{ width: `${pct}%`, background: pctColor, transition: 'width 0.8s cubic-bezier(.22,1,.36,1)' }} />
          </div>
        </div>
      </div>

      {expanded && (
        <div className="mt-3 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
          <p className="text-[11.5px] text-[var(--text2)] mb-3">{path.description}</p>
          <div className="grid grid-cols-2 gap-3 mb-3">
            <div>
              <div className="text-[10.5px] font-bold text-[var(--text3)] uppercase tracking-wider mb-1">Time to Income</div>
              <div className="text-[12px] text-[var(--text)]">{path.time_to_first_income}</div>
            </div>
            <div>
              <div className="text-[10.5px] font-bold text-[var(--text3)] uppercase tracking-wider mb-1">Income Range</div>
              <div className="text-[12px] text-[var(--text)]">{path.income_range}</div>
            </div>
          </div>
          {path.matched_skills.length > 0 && (
            <div className="mb-2">
              <div className="text-[10.5px] font-bold uppercase tracking-wider mb-1" style={{ color: 'var(--green)' }}>Matched Skills</div>
              <div className="flex flex-wrap gap-1">
                {path.matched_skills.slice(0, 10).map(s => (
                  <span key={s} className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--green-bg)', color: 'var(--green)', border: '1px solid var(--green-border)' }}>{s}</span>
                ))}
              </div>
            </div>
          )}
          {path.missing_required.length > 0 && (
            <div className="mb-2">
              <div className="text-[10.5px] font-bold uppercase tracking-wider mb-1" style={{ color: 'var(--red)' }}>Gaps to Fill</div>
              <div className="flex flex-wrap gap-1">
                {path.missing_required.slice(0, 8).map(s => (
                  <span key={s} className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--red-bg)', color: 'var(--red)', border: '1px solid var(--red-border)' }}>{s}</span>
                ))}
              </div>
            </div>
          )}
          {path.starter_actions.length > 0 && (
            <div>
              <div className="text-[10.5px] font-bold text-[var(--text3)] uppercase tracking-wider mb-1">Starter Actions</div>
              <ul className="flex flex-col gap-0.5">
                {path.starter_actions.slice(0, 4).map((a, i) => (
                  <li key={i} className="text-[11px] text-[var(--text2)] flex gap-1.5"><span style={{ color: 'var(--accent)' }}>›</span>{a}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function CareerPaths() {
  const { resumes } = useStore()
  const [resumeText, setResumeText] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [filter, setFilter] = useState('')
  const [paths, setPaths] = useState<PathMatch[]>([])
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')

  const items = Object.values(resumes).filter(r =>
    !filter || r.name?.toLowerCase().includes(filter.toLowerCase())
  )

  const selectResume = (id: string) => {
    const r = resumes[id]
    if (!r) return
    setSelectedId(id); setResumeText(r.text ?? '')
    setPaths([]); setError('')
  }

  const run = async () => {
    if (!resumeText.trim()) { setError('Resume text required.'); return }
    setRunning(true); setError(''); setPaths([])
    try {
      const res = await api<{ matches: PathMatch[] }>('/skill-match', { method: 'POST', body: JSON.stringify({ resume_text: resumeText }) })
      setPaths(res.matches ?? [])
    } catch (e) { setError(e instanceof Error ? e.message : 'Request failed') }
    finally { setRunning(false) }
  }

  return (
    <div>
      <PageHeader title="Career Paths" sub="Discover income paths matched to the candidate's existing skill set." />
      <div className="grid gap-4" style={{ gridTemplateColumns: 'minmax(240px,.7fr) minmax(400px,1.3fr)' }}>
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
            <textarea value={resumeText} onChange={e => setResumeText(e.target.value)} rows={8}
              placeholder="Paste resume or select from library…"
              className="w-full px-3 py-2 text-[11.5px] rounded-md outline-none font-mono resize-y"
              style={inputBase} onFocus={onFocus} onBlur={onBlur} />
            <Button className="w-full mt-3" onClick={run} disabled={running}>
              <TrendingUp size={14} />{running ? 'Matching paths…' : 'Find Career Paths'}
            </Button>
            {error && <div className="text-[11.5px] mt-2" style={{ color: 'var(--red)' }}>{error}</div>}
          </Card>
        </div>

        <div>
          {!paths.length && !running && (
            <Card>
              <div className="text-center py-16 text-[var(--text3)]">
                <div className="text-[32px] mb-3 opacity-30">🗺️</div>
                <div className="text-[13px] font-semibold">No paths yet</div>
                <div className="text-[12px] mt-1">Select a resume and click Find Career Paths.</div>
              </div>
            </Card>
          )}
          {running && (
            <Card>
              <div className="text-center py-16 text-[var(--text3)]">
                <div className="text-[13px] font-semibold">Matching against income paths…</div>
              </div>
            </Card>
          )}
          {paths.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className="text-[12.5px] font-semibold text-[var(--text2)]">Top {paths.length} matches · click to expand</div>
              </div>
              <div className="flex flex-col gap-2">
                {paths.map((p, i) => <PathCard key={p.id} path={p} rank={i + 1} />)}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
