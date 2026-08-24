import { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Trash2, ChevronRight, X } from 'lucide-react'

function PageHeader({ title, sub }: { title: string; sub: string }) {
  return (
    <div className="relative pb-[18px] mb-6" style={{ borderBottom: '1px solid var(--border)' }}>
      <div className="absolute bottom-[-1px] left-0 w-10 h-0.5 rounded-sm bg-[var(--accent)]" />
      <h1 className="font-display text-[20px] font-black tracking-[-0.5px] mb-1">{title}</h1>
      <p className="text-[13px] text-[var(--text2)]">{sub}</p>
    </div>
  )
}

interface RunRow {
  result_id: string
  timestamp: string
  model: string
  resume_name: string
  job_title: string
  match_score: number
  metrics: Record<string, number>
  analysis: {
    missing_critical_skills: string[]
    unmet_responsibilities: string[]
    absent_seniority_signals: string[]
    irrelevant_statements: string[]
    unquantified_statements: string[]
    coach_recommendations: string[]
  }
}

function scoreColor(s: number) {
  if (s >= 70) return 'var(--green)'
  if (s >= 40) return 'var(--amber)'
  return 'var(--red)'
}

function ScoreBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="mb-2">
      <div className="flex justify-between items-center mb-1">
        <span className="text-[11px] text-[var(--text2)]">{label}</span>
        <span className="text-[11.5px] font-bold font-mono" style={{ color: scoreColor(value) }}>{value}%</span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--surface3)' }}>
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${value}%`, background: scoreColor(value) }} />
      </div>
    </div>
  )
}

function Pill({ text, color }: { text: string; color: string }) {
  return (
    <span className="inline-block text-[10.5px] font-semibold px-2 py-0.5 rounded-md mr-1 mb-1" style={{ background: `${color}18`, color, border: `1px solid ${color}44` }}>
      {text}
    </span>
  )
}

function DetailPanel({ row, onClose, onDelete }: { row: RunRow; onClose: () => void; onDelete: (id: string) => void }) {
  const metrics = [
    ['Skills Coverage', row.metrics.skills_coverage],
    ['Experience Depth', row.metrics.experience_depth],
    ['Seniority Calibration', row.metrics.seniority_calibration],
    ['Statement Relevancy', row.metrics.statement_relevancy],
    ['Impact Quantification', row.metrics.impact_quantification],
  ] as [string, number][]

  return (
    <Card className="sticky top-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="text-[13px] font-bold">{row.resume_name || 'Unknown Candidate'}</div>
          <div className="text-[11px] text-[var(--text3)] mt-0.5">{row.job_title || 'Unknown Role'}</div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="danger" size="sm" onClick={() => onDelete(row.result_id)}><Trash2 size={11} /> Delete</Button>
          <button onClick={onClose} className="w-7 h-7 flex items-center justify-center rounded-md hover:bg-[var(--surface3)] transition-colors">
            <X size={14} color="var(--text3)" />
          </button>
        </div>
      </div>

      <div className="flex items-center gap-3 mb-4 p-3 rounded-lg" style={{ background: 'var(--surface2)', border: '1px solid var(--border)' }}>
        <div className="text-[32px] font-black font-display tracking-tight" style={{ color: scoreColor(row.match_score) }}>
          {row.match_score}%
        </div>
        <div>
          <div className="text-[11px] text-[var(--text3)] uppercase tracking-wider font-bold">Match Score</div>
          <div className="text-[10.5px] text-[var(--text3)] mt-0.5 font-mono">{row.model}</div>
        </div>
      </div>

      <div className="mb-4">
        <div className="text-[11px] font-bold text-[var(--text2)] uppercase tracking-wider mb-2">Judge Scores</div>
        {metrics.map(([label, val]) => <ScoreBar key={label} label={label} value={val ?? 0} />)}
      </div>

      {row.analysis.missing_critical_skills?.length > 0 && (
        <div className="mb-3">
          <div className="text-[11px] font-bold text-[var(--text2)] uppercase tracking-wider mb-1.5">Missing Skills</div>
          <div>{row.analysis.missing_critical_skills.slice(0, 8).map(s => <Pill key={s} text={s} color="var(--red)" />)}</div>
        </div>
      )}

      {row.analysis.coach_recommendations?.length > 0 && (
        <div className="mb-3">
          <div className="text-[11px] font-bold text-[var(--text2)] uppercase tracking-wider mb-1.5">Coach Recommendations</div>
          <div className="flex flex-col gap-1.5">
            {(row.analysis.coach_recommendations as unknown as { area?: string; suggestion?: string }[]).slice(0, 3).map((r, i) => (
              <div key={i} className="text-[11.5px] p-2 rounded" style={{ background: 'var(--surface2)', border: '1px solid var(--border)' }}>
                {typeof r === 'string' ? r : (r.suggestion ?? r.area ?? JSON.stringify(r))}
              </div>
            ))}
          </div>
        </div>
      )}

      {row.analysis.unmet_responsibilities?.length > 0 && (
        <div className="mb-3">
          <div className="text-[11px] font-bold text-[var(--text2)] uppercase tracking-wider mb-1.5">Unmet Responsibilities</div>
          <div>{row.analysis.unmet_responsibilities.slice(0, 5).map(s => <Pill key={s} text={s} color="var(--amber)" />)}</div>
        </div>
      )}
    </Card>
  )
}

export function Results() {
  const [rows, setRows] = useState<RunRow[]>([])
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<RunRow | null>(null)

  const load = async () => {
    setLoading(true)
    try { setRows(await api<RunRow[]>('/list_results?limit=100')) }
    catch { /* silent */ }
    finally { setLoading(false) }
  }

  useEffect(() => { void load() }, [])

  const deleteRow = async (id: string) => {
    try {
      await api(`/results/${id}`, { method: 'DELETE' })
      setRows(r => r.filter(x => x.result_id !== id))
      if (selected?.result_id === id) setSelected(null)
    } catch { /* silent */ }
  }

  const fmt = (ts: string) => {
    try { return new Date(ts).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) }
    catch { return ts }
  }

  return (
    <div>
      <PageHeader title="Results" sub="Browse past analysis runs, compare scores, and review judge breakdowns." />
      <div className={`grid gap-4 ${selected ? 'grid-cols-[1fr_320px]' : 'grid-cols-1'}`}>
        <Card>
          <CardHeader>
            <CardTitle>Analysis History</CardTitle>
            <span className="text-[11.5px] text-[var(--text3)]">{rows.length} runs</span>
          </CardHeader>
          {loading ? (
            <div className="text-[13px] text-[var(--text3)] py-6 text-center">Loading…</div>
          ) : rows.length === 0 ? (
            <div className="text-center py-12 text-[var(--text3)]">
              <div className="text-[28px] mb-2 opacity-30">📊</div>
              <div className="text-[13px] font-semibold">No results yet</div>
              <div className="text-[12px] mt-1">Run an analysis to see results here.</div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-[12.5px]">
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    {['Date', 'Candidate', 'Job', 'Score', 'Model', ''].map(h => (
                      <th key={h} className="text-left pb-2 pr-4 text-[10.5px] font-bold text-[var(--text3)] uppercase tracking-wider">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map(row => {
                    const isSel = selected?.result_id === row.result_id
                    return (
                      <tr key={row.result_id} onClick={() => setSelected(isSel ? null : row)}
                        className="cursor-pointer transition-colors"
                        style={{ borderBottom: '1px solid var(--border)', background: isSel ? 'var(--accent-bg)' : 'transparent' }}
                        onMouseEnter={e => { if (!isSel) (e.currentTarget as HTMLElement).style.background = 'var(--surface2)' }}
                        onMouseLeave={e => { if (!isSel) (e.currentTarget as HTMLElement).style.background = 'transparent' }}>
                        <td className="py-2.5 pr-4 text-[var(--text3)] font-mono text-[11px]">{fmt(row.timestamp)}</td>
                        <td className="py-2.5 pr-4 font-semibold">{row.resume_name || <span className="text-[var(--text3)]">—</span>}</td>
                        <td className="py-2.5 pr-4 text-[var(--text2)] max-w-[200px] truncate">{row.job_title || <span className="text-[var(--text3)]">—</span>}</td>
                        <td className="py-2.5 pr-4">
                          <span className="font-black font-display text-[13px]" style={{ color: scoreColor(row.match_score) }}>{row.match_score}%</span>
                        </td>
                        <td className="py-2.5 pr-4 text-[var(--text3)] font-mono text-[10.5px] max-w-[120px] truncate">{row.model}</td>
                        <td className="py-2.5">
                          <ChevronRight size={13} color={isSel ? 'var(--accent)' : 'var(--text3)'} style={{ transform: isSel ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }} />
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        {selected && (
          <DetailPanel row={selected} onClose={() => setSelected(null)} onDelete={deleteRow} />
        )}
      </div>
    </div>
  )
}
