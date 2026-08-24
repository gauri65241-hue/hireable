import { useState } from 'react'
import { useStore } from '@/store'
import { api } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { avatarHue } from '@/lib/utils'
import { Search, Trash2, CheckSquare } from 'lucide-react'

function PageHeader({ title, sub }: { title: string; sub: string }) {
  return (
    <div className="relative pb-[18px] mb-6" style={{ borderBottom: '1px solid var(--border)' }}>
      <div className="absolute bottom-[-1px] left-0 w-10 h-0.5 rounded-sm bg-[var(--accent)]" />
      <h1 className="font-display text-[20px] font-black tracking-[-0.5px] mb-1">{title}</h1>
      <p className="text-[13px] text-[var(--text2)]">{sub}</p>
    </div>
  )
}

const inputCls = 'w-full px-3 py-2 text-[13px] rounded-md outline-none transition-all duration-100 font-body'
const inputBase: React.CSSProperties = { background: 'var(--surface2)', border: '1px solid var(--border2)', color: 'var(--text)' }
const onFocus = (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>) => {
  e.target.style.borderColor = 'var(--accent)'; e.target.style.boxShadow = '0 0 0 3px var(--accent-glow)'
}
const onBlur = (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>) => {
  e.target.style.borderColor = 'var(--border2)'; e.target.style.boxShadow = 'none'
}

type Mode = 'manual' | 'url' | 'paste'

export function Jobs() {
  const { jobs, setJobs, selectedJobs, toggleJob, clearSelections } = useStore()
  const [filter, setFilter] = useState('')
  const [mode, setMode] = useState<Mode>('manual')
  const [form, setForm] = useState({ company: '', title: '', experience_required: '', skills_required: '', description: '', source_url: '' })
  const [urlInput, setUrlInput] = useState('')
  const [pasteInput, setPasteInput] = useState('')
  const [status, setStatus] = useState('')
  const [saving, setSaving] = useState(false)
  const [urlStatus, setUrlStatus] = useState('')

  const reload = async () => {
    try {
      const j = await api<{ job_id: number; company: string; title: string; experience_required: string; skills_required: string; description: string; source_url?: string; applied?: boolean; created_at: string; user_id: number }[]>('/jobs')
      setJobs(Object.fromEntries(j.map(x => [String(x.job_id), x])))
    } catch { /* silent */ }
  }

  const items = Object.values(jobs).filter(j =>
    !filter || j.title?.toLowerCase().includes(filter.toLowerCase()) || j.company?.toLowerCase().includes(filter.toLowerCase())
  )

  const saveJob = async (payload: { company: string; title: string; experience_required: string; skills_required: string; description: string; source_url: string }) => {
    await api('/jobs', { method: 'POST', body: JSON.stringify(payload) })
    await reload()
  }

  const submitManual = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!form.title || !form.company) { setStatus('Title and company are required.'); return }
    setSaving(true); setStatus('')
    try { await saveJob({ ...form }); setForm({ company: '', title: '', experience_required: '', skills_required: '', description: '', source_url: '' }); setStatus('Saved ✓') }
    catch (err) { setStatus(err instanceof Error ? err.message : 'Save failed') }
    finally { setSaving(false) }
  }

  const fetchUrl = async () => {
    if (!urlInput) return
    setUrlStatus('Fetching…')
    try {
      const data = await api<{ company?: string; title?: string; experience_required?: string; skills_required?: string; description?: string; source_url?: string }>('/jobs/fetch', { method: 'POST', body: JSON.stringify({ url: urlInput }) })
      setForm({ company: data.company ?? '', title: data.title ?? '', experience_required: data.experience_required ?? '', skills_required: data.skills_required ?? '', description: data.description ?? '', source_url: data.source_url ?? urlInput })
      setMode('manual'); setUrlStatus('')
    } catch (err) { setUrlStatus(err instanceof Error ? err.message : 'Fetch failed') }
  }

  const parsePaste = async () => {
    if (!pasteInput) return
    setSaving(true)
    try {
      const data = await api<{ company?: string; title?: string; experience_required?: string; skills_required?: string; description?: string }>('/jobs/parse', { method: 'POST', body: JSON.stringify({ text: pasteInput }) })
      setForm({ company: data.company ?? '', title: data.title ?? '', experience_required: data.experience_required ?? '', skills_required: data.skills_required ?? '', description: data.description ?? pasteInput, source_url: '' })
      setMode('manual')
    } catch { setForm(f => ({ ...f, description: pasteInput })); setMode('manual') }
    finally { setSaving(false) }
  }

  const deleteSelected = async () => {
    if (!selectedJobs.size) return
    await Promise.all([...selectedJobs].map(id => api(`/jobs/${id}`, { method: 'DELETE' }).catch(() => {})))
    clearSelections(); await reload()
  }

  const tabs: { id: Mode; label: string }[] = [{ id: 'manual', label: 'Manual' }, { id: 'url', label: 'URL Fetch' }, { id: 'paste', label: 'Quick Paste' }]

  return (
    <div>
      <PageHeader title="Jobs" sub="Browse job postings, add new positions, and manage your job database." />

      <div className="flex items-center justify-between flex-wrap gap-2.5 p-2.5 rounded-[var(--radius)] mb-4" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
        <div className="flex items-center gap-2 px-2.5 rounded-md flex-1 max-w-[320px]" style={{ background: 'var(--surface2)', border: '1px solid var(--border)' }}>
          <Search size={13} color="var(--text3)" />
          <input value={filter} onChange={e => setFilter(e.target.value)} placeholder="Search by company or role…"
            className="border-0 bg-transparent outline-none text-[13px] py-2 flex-1 text-[var(--text)] font-body" />
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" size="sm" onClick={() => Object.keys(jobs).forEach(id => !selectedJobs.has(id) && toggleJob(id))}>
            <CheckSquare size={12} /> Select All
          </Button>
          <Button variant="ghost" size="sm" onClick={clearSelections}>Clear</Button>
          <Button variant="danger" size="sm" onClick={deleteSelected} disabled={!selectedJobs.size}>
            <Trash2 size={12} /> Delete {selectedJobs.size > 0 && `(${selectedJobs.size})`}
          </Button>
        </div>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: 'minmax(300px,.9fr) minmax(360px,1.1fr)' }}>
        <Card style={{ overflow: 'visible' }}>
          <CardHeader><CardTitle>Add Job</CardTitle></CardHeader>

          {/* Tab strip */}
          <div className="flex -mx-[18px] px-[18px] mb-4" style={{ borderBottom: '1px solid var(--border)' }}>
            {tabs.map(t => (
              <button key={t.id} onClick={() => setMode(t.id)}
                className="px-3.5 py-2 text-[12px] font-semibold border-0 cursor-pointer font-body transition-all duration-100 bg-transparent"
                style={{ color: mode === t.id ? 'var(--accent-light)' : 'var(--text2)', borderBottom: `2px solid ${mode === t.id ? 'var(--accent)' : 'transparent'}`, marginBottom: -1 }}>
                {t.label}
              </button>
            ))}
          </div>

          {/* Manual mode */}
          {mode === 'manual' && (
            <form onSubmit={submitManual}>
              {([['company', 'Company *', 'Google'], ['title', 'Job Title *', 'Senior ML Engineer'], ['experience_required', 'Experience Required', '4+ years'], ['skills_required', 'Skills Required', 'Python, FastAPI, RAG']] as [keyof typeof form, string, string][]).map(([key, label, ph]) => (
                <div key={key} className="mb-3.5">
                  <label className="block text-[11px] font-bold text-[var(--text2)] mb-1.5 uppercase tracking-[0.4px]">{label}</label>
                  <input value={form[key]} onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))} placeholder={ph} className={inputCls} style={inputBase} onFocus={onFocus} onBlur={onBlur} />
                </div>
              ))}
              <div className="mb-3.5">
                <label className="block text-[11px] font-bold text-[var(--text2)] mb-1.5 uppercase tracking-[0.4px]">Job Description</label>
                <textarea value={form.description} onChange={e => setForm(f => ({ ...f, description: e.target.value }))} rows={6}
                  placeholder="Paste full job description here…" className={inputCls} style={{ ...inputBase, resize: 'vertical', fontFamily: 'var(--font-mono)', fontSize: 11.5 }} onFocus={onFocus} onBlur={onBlur} />
              </div>
              <div className="mb-3.5">
                <label className="block text-[11px] font-bold text-[var(--text2)] mb-1.5 uppercase tracking-[0.4px]">Source URL</label>
                <input value={form.source_url} onChange={e => setForm(f => ({ ...f, source_url: e.target.value }))} placeholder="https://…" className={inputCls} style={inputBase} onFocus={onFocus} onBlur={onBlur} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[11.5px]" style={{ color: status.includes('✓') ? 'var(--green)' : 'var(--red)' }}>{status}</span>
                <Button type="submit" disabled={saving}>{saving ? 'Saving…' : 'Save Job'}</Button>
              </div>
            </form>
          )}

          {/* URL mode */}
          {mode === 'url' && (
            <div>
              <div className="mb-3.5">
                <label className="block text-[11px] font-bold text-[var(--text2)] mb-1.5 uppercase tracking-[0.4px]">Job Posting URL</label>
                <div className="flex gap-2">
                  <input value={urlInput} onChange={e => setUrlInput(e.target.value)} placeholder="https://jobs.greenhouse.io/…" className={inputCls} style={{ ...inputBase, flex: 1 }} onFocus={onFocus} onBlur={onBlur} />
                  <Button onClick={fetchUrl} disabled={saving}>Fetch</Button>
                </div>
                {urlStatus && <div className="text-[11.5px] mt-2" style={{ color: urlStatus.includes('ailed') ? 'var(--red)' : 'var(--text3)' }}>{urlStatus}</div>}
                <div className="text-[11px] mt-2" style={{ color: 'var(--amber)' }}>⚠ LinkedIn requires login — use Quick Paste for LinkedIn JDs</div>
              </div>
            </div>
          )}

          {/* Paste mode */}
          {mode === 'paste' && (
            <div>
              <div className="mb-3.5">
                <label className="block text-[11px] font-bold text-[var(--text2)] mb-1.5 uppercase tracking-[0.4px]">Paste Raw Job Description</label>
                <textarea value={pasteInput} onChange={e => setPasteInput(e.target.value)} rows={10}
                  placeholder="Copy the full job posting text from LinkedIn or any site and paste here…"
                  className={inputCls} style={{ ...inputBase, resize: 'vertical', fontFamily: 'var(--font-mono)', fontSize: 11.5 }} onFocus={onFocus} onBlur={onBlur} />
              </div>
              <Button onClick={parsePaste} disabled={saving}>{saving ? 'Parsing…' : 'Parse & Preview'}</Button>
            </div>
          )}
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Available Jobs</CardTitle>
            <span className="text-[11.5px] text-[var(--text3)]">{selectedJobs.size} selected</span>
          </CardHeader>
          <div className="flex flex-col gap-1.5 max-h-[520px] overflow-y-auto">
            {items.length === 0 ? (
              <div className="text-center py-10 text-[var(--text3)]">
                <div className="text-[28px] mb-2 opacity-30">💼</div>
                <div className="text-[12.5px] font-semibold">No jobs yet</div>
              </div>
            ) : items.map(j => {
              const sel = selectedJobs.has(String(j.job_id))
              const hue = avatarHue(j.job_id, 83)
              const co = (j.company || '?')[0].toUpperCase()
              return (
                <div key={j.job_id} onClick={() => toggleJob(String(j.job_id))}
                  className="flex items-start gap-2.5 p-2.5 rounded-md cursor-pointer transition-all duration-100"
                  style={{ border: `1px solid ${sel ? 'var(--accent-border)' : 'var(--border)'}`, background: sel ? 'var(--accent-bg)' : 'var(--surface2)' }}>
                  <div className="w-7 h-7 rounded-md flex items-center justify-center text-[11px] font-black font-display flex-shrink-0 mt-0.5"
                    style={{ background: `hsla(${hue},55%,50%,.15)`, color: `hsla(${hue},65%,70%,1)`, boxShadow: `0 0 0 1px hsla(${hue},55%,50%,.25)` }}>
                    {co}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5 text-[12.5px] font-semibold flex-wrap">
                      {j.title}
                      {sel && <span className="text-[9px] font-bold px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-bg)', color: 'var(--accent-light)', border: '1px solid var(--accent-border)' }}>SELECTED</span>}
                      {j.applied && <span className="text-[9px] font-bold px-1.5 py-0.5 rounded" style={{ background: 'var(--green-bg)', color: 'var(--green)', border: '1px solid var(--green-border)' }}>✓ APPLIED</span>}
                    </div>
                    <div className="text-[11px] text-[var(--text3)] mt-0.5">#{j.job_id} · {j.company}</div>
                  </div>
                </div>
              )
            })}
          </div>
        </Card>
      </div>
    </div>
  )
}
