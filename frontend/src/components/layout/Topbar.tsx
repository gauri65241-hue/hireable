import { useStore } from '@/store'
import { RefreshCw, ExternalLink, Sun, Moon } from 'lucide-react'

const TITLES: Record<string, string> = {
  health:        'System Health',
  resumes:       'Resumes',
  jobs:          'Jobs',
  analyzer:      'Analyzer',
  results:       'Results',
  'ats-checker': 'ATS Checker',
  'career-paths':'Career Paths',
  interview:     'Interview Prep',
  benchmark:     'Model Benchmark',
}

export function Topbar({ onRefresh }: { onRefresh?: () => void }) {
  const { currentPage, theme, toggleTheme } = useStore()
  return (
    <header className="sticky top-0 z-50 flex items-center gap-3.5 px-6" style={{ height: 'var(--topbar)', background: 'var(--surface)', borderBottom: '1px solid var(--border)', backdropFilter: 'blur(8px)' }}>
      <div className="font-display text-[14px] font-bold tracking-[-0.2px] text-[var(--text)]">{TITLES[currentPage] ?? currentPage}</div>
      <div className="ml-auto flex items-center gap-2">
        <button onClick={toggleTheme} title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          className="w-[30px] h-[30px] flex items-center justify-center rounded-md cursor-pointer transition-all duration-100"
          style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text3)' }}
          onMouseOver={e => { (e.currentTarget as HTMLElement).style.color = 'var(--text)'; (e.currentTarget as HTMLElement).style.borderColor = 'var(--border2)' }}
          onMouseOut={e => { (e.currentTarget as HTMLElement).style.color = 'var(--text3)'; (e.currentTarget as HTMLElement).style.borderColor = 'var(--border)' }}>
          {theme === 'dark' ? <Sun size={13} /> : <Moon size={13} />}
        </button>
        <button onClick={onRefresh} className="w-[30px] h-[30px] flex items-center justify-center rounded-md cursor-pointer transition-all duration-100" style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text3)' }}
          onMouseOver={e => { (e.currentTarget as HTMLElement).style.color = 'var(--text)'; (e.currentTarget as HTMLElement).style.borderColor = 'var(--border2)' }}
          onMouseOut={e => { (e.currentTarget as HTMLElement).style.color = 'var(--text3)'; (e.currentTarget as HTMLElement).style.borderColor = 'var(--border)' }}>
          <RefreshCw size={13} />
        </button>
        <a href="/docs" target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 px-3 py-1 text-[12px] font-semibold text-white rounded-md no-underline transition-opacity hover:opacity-90" style={{ background: 'var(--accent)' }}>
          API Docs <ExternalLink size={11} />
        </a>
      </div>
    </header>
  )
}
