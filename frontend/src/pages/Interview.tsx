import { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { ChevronDown, ChevronUp, MessageSquare } from 'lucide-react'

function PageHeader({ title, sub }: { title: string; sub: string }) {
  return (
    <div className="relative pb-[18px] mb-6" style={{ borderBottom: '1px solid var(--border)' }}>
      <div className="absolute bottom-[-1px] left-0 w-10 h-0.5 rounded-sm bg-[var(--accent)]" />
      <h1 className="font-display text-[20px] font-black tracking-[-0.5px] mb-1">{title}</h1>
      <p className="text-[13px] text-[var(--text2)]">{sub}</p>
    </div>
  )
}

interface RunRow { result_id: string; timestamp: string; resume_name: string; job_title: string; match_score: number; model: string }
interface Question { gap: string; question: string; model_answer: string; difficulty: 'easy' | 'medium' | 'hard'; type: 'technical' | 'behavioral' | 'situational' }

function difficultyColor(d: string) {
  if (d === 'easy') return 'var(--green)'
  if (d === 'medium') return 'var(--amber)'
  return 'var(--red)'
}
function typeColor(t: string) {
  if (t === 'technical') return 'var(--accent)'
  if (t === 'behavioral') return 'var(--amber)'
  return 'var(--green)'
}

function QuestionCard({ q, idx }: { q: Question; idx: number }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="rounded-[var(--radius)] overflow-hidden" style={{ border: '1px solid var(--border)' }}>
      <div className="flex items-start gap-3 p-3.5 cursor-pointer transition-colors hover:bg-[var(--surface2)]"
        style={{ background: 'var(--surface)' }} onClick={() => setOpen(o => !o)}>
        <div className="w-6 h-6 rounded flex items-center justify-center text-[10px] font-black flex-shrink-0 mt-0.5"
          style={{ background: 'var(--accent-bg)', color: 'var(--accent-light)', border: '1px solid var(--accent-border)' }}>
          {idx}
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-[12.5px] font-semibold mb-1">{q.question}</div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[9.5px] font-bold px-2 py-0.5 rounded-full" style={{ background: `${difficultyColor(q.difficulty)}18`, color: difficultyColor(q.difficulty), border: `1px solid ${difficultyColor(q.difficulty)}44` }}>{q.difficulty}</span>
            <span className="text-[9.5px] font-bold px-2 py-0.5 rounded-full" style={{ background: `${typeColor(q.type)}18`, color: typeColor(q.type), border: `1px solid ${typeColor(q.type)}44` }}>{q.type}</span>
            <span className="text-[10px] text-[var(--text3)]">Gap: {q.gap}</span>
          </div>
        </div>
        <div className="flex-shrink-0 mt-1">
          {open ? <ChevronUp size={14} color="var(--text3)" /> : <ChevronDown size={14} color="var(--text3)" />}
        </div>
      </div>
      {open && (
        <div className="px-4 pb-4 pt-2" style={{ borderTop: '1px solid var(--border)', background: 'var(--surface2)' }}>
          <div className="text-[10.5px] font-bold text-[var(--text3)] uppercase tracking-wider mb-1.5">Model Answer</div>
          <div className="text-[12px] text-[var(--text2)] leading-relaxed whitespace-pre-wrap">{q.model_answer}</div>
        </div>
      )}
    </div>
  )
}

export function Interview() {
  const [rows, setRows] = useState<RunRow[]>([])
  const [loadingRows, setLoadingRows] = useState(true)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [model, setModel] = useState('cogito:14b')
  const [questions, setQuestions] = useState<Question[]>([])
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    api<RunRow[]>('/list_results?limit=50')
      .then(r => setRows(r))
      .catch(() => {})
      .finally(() => setLoadingRows(false))
  }, [])

  const run = async () => {
    if (!selectedId) { setError('Select a result first.'); return }
    setRunning(true); setError(''); setQuestions([])
    try {
      const res = await api<{ questions: Question[] }>('/interview_prep', { method: 'POST', body: JSON.stringify({ result_id: selectedId, model }) })
      setQuestions(res.questions ?? [])
    } catch (e) { setError(e instanceof Error ? e.message : 'Request failed') }
    finally { setRunning(false) }
  }

  const fmt = (ts: string) => {
    try { return new Date(ts).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) }
    catch { return ts }
  }

  const scoreColor = (s: number) => s >= 70 ? 'var(--green)' : s >= 40 ? 'var(--amber)' : 'var(--red)'

  const byType = (type: string) => questions.filter(q => q.type === type)

  return (
    <div>
      <PageHeader title="Interview Prep" sub="Generate tailored questions from gap analysis so candidates can rehearse weak spots." />
      <div className="grid gap-4" style={{ gridTemplateColumns: 'minmax(280px,.75fr) minmax(420px,1.25fr)' }}>
        <div className="flex flex-col gap-4">
          <Card>
            <CardHeader><CardTitle>Select Analysis Run</CardTitle></CardHeader>
            {loadingRows ? (
              <div className="text-[12px] text-[var(--text3)] py-4 text-center">Loading…</div>
            ) : rows.length === 0 ? (
              <div className="text-[12px] text-[var(--text3)] py-4 text-center">No results yet. Run an analysis first.</div>
            ) : (
              <div className="flex flex-col gap-1.5 max-h-[320px] overflow-y-auto">
                {rows.map(r => {
                  const sel = selectedId === r.result_id
                  return (
                    <div key={r.result_id} onClick={() => { setSelectedId(r.result_id); setQuestions([]); setError('') }}
                      className="p-2.5 rounded-md cursor-pointer transition-all duration-100"
                      style={{ border: `1px solid ${sel ? 'var(--accent-border)' : 'var(--border)'}`, background: sel ? 'var(--accent-bg)' : 'var(--surface2)' }}>
                      <div className="flex items-center justify-between">
                        <span className="text-[12px] font-semibold">{r.resume_name || 'Unknown'}</span>
                        <span className="text-[12px] font-black font-display" style={{ color: scoreColor(r.match_score) }}>{r.match_score}%</span>
                      </div>
                      <div className="text-[10.5px] text-[var(--text3)] mt-0.5">{r.job_title || 'Unknown role'} · {fmt(r.timestamp)}</div>
                    </div>
                  )
                })}
              </div>
            )}
          </Card>

          <Card>
            <CardHeader><CardTitle>Model</CardTitle></CardHeader>
            <select value={model} onChange={e => setModel(e.target.value)}
              className="w-full px-3 py-2 text-[12.5px] rounded-md outline-none font-mono"
              style={{ background: 'var(--surface2)', border: '1px solid var(--border2)', color: 'var(--text)' }}>
              <option value="cogito:14b">cogito:14b</option>
              <option value="qwen2.5-coder:7b">qwen2.5-coder:7b</option>
              <option value="mistral:7b">mistral:7b</option>
            </select>
            <Button className="w-full mt-3" onClick={run} disabled={running || !selectedId}>
              <MessageSquare size={14} />{running ? 'Generating…' : 'Generate Questions'}
            </Button>
            {error && <div className="text-[11.5px] mt-2" style={{ color: 'var(--red)' }}>{error}</div>}
            {running && <div className="text-[11px] text-[var(--text3)] mt-2 text-center">LLM generating 8–12 targeted questions…</div>}
          </Card>
        </div>

        <div>
          {!questions.length && !running && (
            <Card>
              <div className="text-center py-16 text-[var(--text3)]">
                <div className="text-[32px] mb-3 opacity-30">💬</div>
                <div className="text-[13px] font-semibold">No questions yet</div>
                <div className="text-[12px] mt-1">Select a run and generate prep questions.</div>
              </div>
            </Card>
          )}
          {running && (
            <Card>
              <div className="text-center py-16 text-[var(--text3)]">
                <div className="text-[13px] font-semibold">Generating interview questions…</div>
                <div className="text-[12px] mt-1">This takes 20–60s depending on model.</div>
              </div>
            </Card>
          )}
          {questions.length > 0 && (
            <div className="flex flex-col gap-5">
              <div className="flex items-center justify-between">
                <div className="text-[13px] font-semibold">{questions.length} questions · click to reveal answers</div>
              </div>
              {(['technical', 'behavioral', 'situational'] as const).map(type => {
                const qs = byType(type)
                if (!qs.length) return null
                return (
                  <div key={type}>
                    <div className="text-[11px] font-black uppercase tracking-wider mb-2 px-0.5" style={{ color: typeColor(type) }}>
                      {type} ({qs.length})
                    </div>
                    <div className="flex flex-col gap-2">
                      {qs.map((q, i) => <QuestionCard key={i} q={q} idx={questions.indexOf(q) + 1} />)}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
