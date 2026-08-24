import { useStore } from '@/store'
import { useAuth } from '@/hooks/useAuth'
import { cn } from '@/lib/utils'
import {
  Activity, FileText, Briefcase, BarChart3, ClipboardList,
  CheckSquare, GitBranch, Mic, TrendingUp, LogOut,
} from 'lucide-react'

const NAV = [
  { id: 'health',        label: 'System Health',  Icon: Activity,       group: 'Main' },
  { id: 'resumes',       label: 'Resumes',         Icon: FileText,       group: 'Main' },
  { id: 'jobs',          label: 'Jobs',            Icon: Briefcase,      group: 'Main' },
  { id: 'analyzer',      label: 'Analyzer',        Icon: BarChart3,      group: 'Main' },
  { id: 'results',       label: 'Results',         Icon: ClipboardList,  group: 'Main' },
  { id: 'ats-checker',   label: 'ATS Checker',     Icon: CheckSquare,    group: 'Tools' },
  { id: 'career-paths',  label: 'Career Paths',    Icon: GitBranch,      group: 'Tools' },
  { id: 'interview',     label: 'Interview Prep',  Icon: Mic,            group: 'Tools' },
  { id: 'benchmark',     label: 'Benchmark',       Icon: TrendingUp,     group: 'Tools' },
]

export function Sidebar() {
  const { currentPage, setCurrentPage } = useStore()
  const { isAuthenticated, getEmail, logout } = useAuth()
  const email = getEmail()
  const initial = email ? email[0].toUpperCase() : '?'

  return (
    <aside className="fixed top-0 left-0 h-screen z-[100] flex flex-col" style={{ width: 'var(--sidebar)', background: 'var(--surface)', borderRight: '1px solid var(--border)' }}>

      {/* Logo */}
      <div className="px-4 py-5 flex items-center gap-2.5" style={{ borderBottom: '1px solid var(--border)' }}>
        <div className="w-8 h-8 rounded-lg flex items-center justify-center text-[15px] flex-shrink-0" style={{ background: 'linear-gradient(135deg,var(--accent) 0%,#8B5CF6 100%)', boxShadow: '0 0 0 1px rgba(99,102,241,.3),0 4px 12px rgba(99,102,241,.25)' }}>🎯</div>
        <div>
          <div className="font-display text-[15px] font-black tracking-[-0.4px]" style={{ background: 'linear-gradient(120deg,var(--text) 30%,var(--accent-light))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>hireable</div>
          <div className="text-[10px] text-[var(--text3)] tracking-[0.2px]">AI Resume Scorer</div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto p-2">
        {(['Main', 'Tools'] as const).map(group => (
          <div key={group}>
            <div className="text-[9.5px] font-bold text-[var(--text3)] uppercase tracking-[1px] px-2 pt-3 pb-1.5">{group}</div>
            {NAV.filter(n => n.group === group).map(({ id, label, Icon }) => {
              const active = currentPage === id
              return (
                <button key={id} onClick={() => setCurrentPage(id)}
                  className={cn(
                    'w-full flex items-center gap-2.5 px-2.5 py-[7px] rounded-md text-[13px] font-medium transition-all duration-100 text-left border-0 font-body cursor-pointer relative',
                    active
                      ? 'bg-[var(--accent-bg)] text-[var(--accent-light)] font-semibold'
                      : 'text-[var(--text2)] bg-transparent hover:bg-[var(--surface2)] hover:text-[var(--text)]',
                  )}>
                  {active && <span className="absolute left-0 top-[20%] bottom-[20%] w-[2.5px] bg-[var(--accent)] rounded-r-sm" />}
                  <Icon size={15} className={active ? 'opacity-100' : 'opacity-60'} />
                  {label}
                </button>
              )
            })}
          </div>
        ))}
      </nav>

      {/* User footer */}
      <div className="p-2.5" style={{ borderTop: '1px solid var(--border)' }}>
        <div className="flex items-center gap-2.5 px-2.5 py-2 rounded-lg" style={{ border: '1px solid var(--border)', background: 'var(--surface2)' }}>
          <div className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 font-display font-bold text-[11px] text-white" style={{ background: 'linear-gradient(135deg,var(--accent) 0%,#8B5CF6 100%)', boxShadow: '0 0 0 1px rgba(99,102,241,.3)' }}>
            {initial}
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-[11.5px] font-semibold text-[var(--text)] truncate">{isAuthenticated ? (email || 'Signed in') : 'Not signed in'}</div>
            <div className="text-[10px] text-[var(--text3)]">{isAuthenticated ? 'Active session' : 'Sign in to continue'}</div>
          </div>
          {isAuthenticated && (
            <button onClick={logout} title="Sign out" className="w-6 h-6 flex items-center justify-center rounded cursor-pointer transition-all duration-100 hover:bg-[var(--red-bg)] hover:text-[var(--red)]" style={{ border: '1px solid var(--border)', background: 'transparent', color: 'var(--text3)' }}>
              <LogOut size={11} />
            </button>
          )}
        </div>
      </div>
    </aside>
  )
}
