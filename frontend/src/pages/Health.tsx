import { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Shield, FileText, Briefcase, Cpu } from 'lucide-react'

interface HealthData {
  status: string
  version: string
  timestamp: string
  concurrency: { max_concurrent_analyses: number; analyses_in_flight: number }
  services: {
    ollama: string; ollama_detail: string; available_models: string[]
    openrouter: string; openrouter_detail: string; openrouter_model: string; openrouter_fast_model: string
    database: string; database_detail: string
  }
}

interface HealthProps { resumeCount: number; jobCount: number }

function PageHeader({ title, sub }: { title: string; sub: string }) {
  return (
    <div className="relative pb-[18px] mb-6" style={{ borderBottom: '1px solid var(--border)' }}>
      <div className="absolute bottom-[-1px] left-0 w-10 h-0.5 rounded-sm bg-[var(--accent)]" />
      <h1 className="font-display text-[20px] font-black tracking-[-0.5px] mb-1">{title}</h1>
      <p className="text-[13px] text-[var(--text2)]">{sub}</p>
    </div>
  )
}

export function Health({ resumeCount, jobCount }: HealthProps) {
  const [health, setHealth] = useState<HealthData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api<HealthData>('/health').then(setHealth).catch(console.error).finally(() => setLoading(false))
  }, [])

  const ollamaOk = health?.services.ollama === 'connected'
  const openrouterOk = health?.services.openrouter === 'connected'
  const dbOk = health?.services.database === 'connected'
  const overall = health?.status === 'healthy'

  const stats = [
    { Icon: Shield,   color: overall ? 'green' : 'red',    iconColor: overall ? 'var(--green)' : 'var(--red)',   value: loading ? '…' : (overall ? 'Healthy' : 'Degraded'), label: 'API Status',   sub: health?.version ? `v${health.version}` : '' },
    { Icon: FileText, color: 'green',  iconColor: 'var(--green)',  value: resumeCount || '-', label: 'Resumes',      sub: 'In database' },
    { Icon: Briefcase,color: 'amber',  iconColor: 'var(--amber)',  value: jobCount || '-',    label: 'Jobs',         sub: 'In database' },
    { Icon: Cpu,      color: 'accent', iconColor: 'var(--accent-light)', value: loading ? '…' : (openrouterOk || ollamaOk ? 'Online' : 'Offline'), label: 'LLM Service', sub: openrouterOk ? 'OpenRouter' : (ollamaOk ? 'Ollama' : 'No LLM') },
  ]

  const services = health ? [
    { name: 'OpenRouter',      ok: openrouterOk, detail: openrouterOk ? 'auto-routing active' : 'not configured', sub: openrouterOk ? `default: ${health.services.openrouter_model} · fast: ${health.services.openrouter_fast_model}` : '' },
    { name: 'Ollama LLM',      ok: ollamaOk, detail: health.services.ollama_detail,   sub: health.services.available_models.join(', ') },
    { name: 'Database',        ok: dbOk,     detail: health.services.database_detail,  sub: '' },
    { name: 'Analysis Queue',  ok: true,     detail: `${health.concurrency.analyses_in_flight} in-flight / ${health.concurrency.max_concurrent_analyses} max`, sub: '' },
  ] : []

  return (
    <div>
      <PageHeader title="System Health" sub="Monitor API status, database connectivity, and LLM service health." />

      {/* Stat cards */}
      <div className="grid grid-cols-4 gap-3 mb-5">
        {stats.map(({ Icon, color, iconColor, value, label, sub }) => (
          <div key={label} className="relative overflow-hidden rounded-[var(--radius)] p-4 transition-all duration-150 hover:border-[var(--border2)]"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <div className="absolute inset-0 pointer-events-none" style={{ background: 'linear-gradient(135deg,rgba(99,102,241,.04) 0%,transparent 60%)' }} />
            {/* Icon badge */}
            <div className="w-[34px] h-[34px] rounded-[8px] flex items-center justify-center mb-2.5" style={{ background: `var(--${color}-bg, var(--accent-bg))`, boxShadow: `0 0 0 1px var(--${color}-border, var(--accent-border))` }}>
              <Icon size={16} color={iconColor} />
            </div>
            <div className="font-display font-black tracking-[-0.5px] mb-0.5" style={{ fontSize: typeof value === 'string' && value.length > 5 ? 18 : 26 }}>{value}</div>
            <div className="text-[10.5px] text-[var(--text3)] uppercase tracking-[0.6px] font-semibold">{label}</div>
            {sub && <div className="text-[11px] font-semibold mt-1.5" style={{ color: iconColor }}>{sub}</div>}
          </div>
        ))}
      </div>

      {/* Service details */}
      <Card>
        <CardHeader><CardTitle>Service Details</CardTitle></CardHeader>
        {loading ? (
          <div className="text-[var(--text3)] text-[13px]">Loading service information…</div>
        ) : health ? (
          <div>
            {services.map(({ name, ok, detail, sub }, i) => (
              <div key={name} className="flex items-center gap-3 py-2.5" style={{ borderBottom: i < services.length - 1 ? '1px solid var(--border)' : 'none' }}>
                <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 text-[12px] font-bold" style={{ background: ok ? 'var(--green-bg)' : 'var(--red-bg)', color: ok ? 'var(--green)' : 'var(--red)' }}>
                  {ok ? '✓' : '✗'}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-[12.5px] font-semibold">{name}</div>
                  {sub && <div className="text-[11px] text-[var(--text3)] mt-0.5 font-mono truncate">{sub}</div>}
                </div>
                <div className="text-[11px] text-[var(--text3)] font-mono">{detail}</div>
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10.5px] font-bold" style={{ background: ok ? 'var(--green-bg)' : 'var(--red-bg)', color: ok ? 'var(--green)' : 'var(--red)', border: `1px solid ${ok ? 'var(--green-border)' : 'var(--red-border)'}` }}>
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: ok ? 'var(--green)' : 'var(--red)' }} />
                  {ok ? 'Online' : 'Offline'}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-[var(--text3)] text-[13px]">Failed to load health data.</div>
        )}
      </Card>
    </div>
  )
}
