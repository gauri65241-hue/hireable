import { useState, useRef } from 'react'
import { useStore } from '@/store'
import { api } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { initials, avatarHue } from '@/lib/utils'
import { Search, Upload, Trash2, CheckSquare } from 'lucide-react'

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

export function Resumes() {
  const { resumes, setResumes, selectedResumes, toggleResume, clearSelections } = useStore()
  const [filter, setFilter] = useState('')
  const [form, setForm] = useState({ name: '', title: '', experience_years: '', skills: '', text: '' })
  const [status, setStatus] = useState('')
  const [saving, setSaving] = useState(false)
  const [pdfStatus, setPdfStatus] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  const reload = async () => {
    try {
      const r = await api<{ resume_id: number; name: string; title: string; years_of_experience: number; skills: string; text: string; created_at: string; user_id: number }[]>('/resumes')
      setResumes(Object.fromEntries(r.map(x => [String(x.resume_id), x])))
    } catch { /* silent */ }
  }

  const items = Object.values(resumes).filter(r =>
    !filter || r.name?.toLowerCase().includes(filter.toLowerCase()) || r.title?.toLowerCase().includes(filter.toLowerCase())
  )

  const save = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!form.name || !form.text) { setStatus('Name and resume text are required.'); return }
    setSaving(true); setStatus('')
    try {
      await api('/resumes', { method: 'POST', body: JSON.stringify({ name: form.name, title: form.title, skills: form.skills, text: form.text, experience_years: form.experience_years }) })
      setForm({ name: '', title: '', experience_years: '', skills: '', text: '' })
      setStatus('Saved ✓'); await reload()
    } catch (err) { setStatus(err instanceof Error ? err.message : 'Save failed') }
    finally { setSaving(false) }
  }

  const uploadPdf = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return
    setPdfStatus('Parsing PDF…')
    try {
      const fd = new FormData(); fd.append('file', file)
      const token = (JSON.parse(localStorage.getItem('hireable-store') || '{}') as { state?: { token?: string } }).state?.token
      const res = await fetch('/pdf/parse-resume', { method: 'POST', headers: token ? { Authorization: `Bearer ${token}` } : {}, body: fd })
      const data = await res.json() as { text?: string }
      if (data.text) { setForm(f => ({ ...f, text: data.text! })); setPdfStatus('PDF parsed ✓') }
      else setPdfStatus('Parse failed')
    } catch { setPdfStatus('Upload error') }
  }

  const deleteSelected = async () => {
    if (!selectedResumes.size) return
    await Promise.all([...selectedResumes].map(id => api(`/resumes/${id}`, { method: 'DELETE' }).catch(() => {})))
    clearSelections(); await reload()
  }

  return (
    <div>
      <PageHeader title="Resumes" sub="Browse candidate profiles, upload new resumes, and manage your talent pool." />

      <div className="flex items-center justify-between flex-wrap gap-2.5 p-2.5 rounded-[var(--radius)] mb-4" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
        <div className="flex items-center gap-2 px-2.5 rounded-md flex-1 max-w-[320px] transition-all duration-100" style={{ background: 'var(--surface2)', border: '1px solid var(--border)' }}>
          <Search size={13} color="var(--text3)" />
          <input value={filter} onChange={e => setFilter(e.target.value)} placeholder="Search by name or title…"
            className="border-0 bg-transparent outline-none text-[13px] py-2 flex-1 text-[var(--text)] font-body" />
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" size="sm" onClick={() => Object.keys(resumes).forEach(id => !selectedResumes.has(id) && toggleResume(id))}>
            <CheckSquare size={12} /> Select All
          </Button>
          <Button variant="ghost" size="sm" onClick={clearSelections}>Clear</Button>
          <Button variant="danger" size="sm" onClick={deleteSelected} disabled={!selectedResumes.size}>
            <Trash2 size={12} /> Delete {selectedResumes.size > 0 && `(${selectedResumes.size})`}
          </Button>
        </div>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: 'minmax(300px,.9fr) minmax(360px,1.1fr)' }}>
        <Card>
          <CardHeader><CardTitle>Add Resume</CardTitle></CardHeader>
          <form onSubmit={save}>
            {([['name', 'Candidate Name *', 'Jane Doe'], ['title', 'Current Title', 'Senior ML Engineer'], ['experience_years', 'Years of Experience', '4'], ['skills', 'Core Skills', 'Python, PyTorch, AWS']] as [keyof typeof form, string, string][]).map(([key, label, ph]) => (
              <div key={key} className="mb-3.5">
                <label className="block text-[11px] font-bold text-[var(--text2)] mb-1.5 uppercase tracking-[0.4px]">{label}</label>
                <input value={form[key]} onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))} placeholder={ph} className={inputCls} style={inputBase} onFocus={onFocus} onBlur={onBlur} />
              </div>
            ))}
            <div className="mb-3.5">
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-[11px] font-bold text-[var(--text2)] uppercase tracking-[0.4px]">Resume Text *</label>
                <div className="flex items-center gap-2">
                  <input ref={fileRef} type="file" accept=".pdf" className="hidden" onChange={uploadPdf} />
                  <Button type="button" variant="secondary" size="sm" onClick={() => fileRef.current?.click()}><Upload size={11} /> Upload PDF</Button>
                  {pdfStatus && <span className="text-[11px] text-[var(--text3)]">{pdfStatus}</span>}
                </div>
              </div>
              <textarea value={form.text} onChange={e => setForm(f => ({ ...f, text: e.target.value }))} rows={8}
                placeholder="Paste full resume text here, or upload a PDF above…"
                className={inputCls} style={{ ...inputBase, resize: 'vertical', fontFamily: 'var(--font-mono)', fontSize: 11.5 }}
                onFocus={onFocus} onBlur={onBlur} />
            </div>
            <div className="flex items-center justify-between mt-1">
              <span className="text-[11.5px]" style={{ color: status.includes('✓') ? 'var(--green)' : 'var(--red)' }}>{status}</span>
              <Button type="submit" disabled={saving}>{saving ? 'Saving…' : 'Save Resume'}</Button>
            </div>
          </form>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Available Resumes</CardTitle>
            <span className="text-[11.5px] text-[var(--text3)]">{selectedResumes.size} selected</span>
          </CardHeader>
          <div className="flex flex-col gap-1.5 max-h-[520px] overflow-y-auto">
            {items.length === 0 ? (
              <div className="text-center py-10 text-[var(--text3)]">
                <div className="text-[28px] mb-2 opacity-30">📄</div>
                <div className="text-[12.5px] font-semibold">No resumes yet</div>
              </div>
            ) : items.map(r => {
              const sel = selectedResumes.has(String(r.resume_id))
              const hue = avatarHue(r.resume_id)
              return (
                <div key={r.resume_id} onClick={() => toggleResume(String(r.resume_id))}
                  className="flex items-start gap-2.5 p-2.5 rounded-md cursor-pointer transition-all duration-100"
                  style={{ border: `1px solid ${sel ? 'var(--accent-border)' : 'var(--border)'}`, background: sel ? 'var(--accent-bg)' : 'var(--surface2)' }}>
                  <div className="w-7 h-7 rounded-md flex items-center justify-center text-[10.5px] font-black font-display flex-shrink-0 mt-0.5"
                    style={{ background: `hsla(${hue},55%,50%,.15)`, color: `hsla(${hue},65%,70%,1)`, boxShadow: `0 0 0 1px hsla(${hue},55%,50%,.25)` }}>
                    {initials(r.name)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5 text-[12.5px] font-semibold">
                      {r.name}
                      {sel && <span className="text-[9px] font-bold px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-bg)', color: 'var(--accent-light)', border: '1px solid var(--accent-border)' }}>SELECTED</span>}
                    </div>
                    <div className="text-[11px] text-[var(--text3)] mt-0.5">#{r.resume_id} · {r.title}</div>
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
