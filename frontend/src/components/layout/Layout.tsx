import { Sidebar } from './Sidebar'
import { Topbar } from './Topbar'

interface LayoutProps {
  children: React.ReactNode
  onRefresh?: () => void
}

export function Layout({ children, onRefresh }: LayoutProps) {
  return (
    <div className="flex min-h-screen" style={{ background: 'var(--bg)' }}>
      <Sidebar />
      <main className="flex flex-col flex-1 min-h-screen" style={{ marginLeft: 'var(--sidebar)' }}>
        <Topbar onRefresh={onRefresh} />
        <div className="p-6 flex-1" style={{ maxWidth: 1200 }}>
          {children}
        </div>
      </main>
    </div>
  )
}
