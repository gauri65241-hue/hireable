import { useState } from 'react'
import { useAuth } from '@/hooks/useAuth'
import { api } from '@/lib/api'

export function AuthOverlay({ visible }: { visible: boolean }) {
  const [tab, setTab] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()

  const submit = async () => {
    setError(''); setLoading(true)
    try {
      const path = tab === 'login' ? '/auth/login' : '/auth/register'
      const data = await api<{ access_token: string }>(path, {
        method: 'POST',
        body: JSON.stringify({ email, password }),
      })
      login(data.access_token)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Authentication failed')
    } finally { setLoading(false) }
  }

  if (!visible) return null

  const inputStyle: React.CSSProperties = {
    width: '100%', padding: '9px 12px', borderRadius: '7px',
    border: '1px solid var(--border2)', background: 'var(--surface2)',
    color: 'var(--text)', fontSize: '13px', fontFamily: 'var(--font-body)',
    outline: 'none', boxSizing: 'border-box', transition: 'border-color .12s, box-shadow .12s',
  }

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 10000, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(8,8,9,.96)', backdropFilter: 'blur(12px)' }}>
      <div style={{ width: '100%', maxWidth: 380, padding: '0 20px' }}>

        {/* Logo */}
        <div style={{ textAlign: 'center', marginBottom: 28 }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 11, marginBottom: 10 }}>
            <div style={{ width: 36, height: 36, borderRadius: 9, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18, background: 'linear-gradient(135deg,var(--accent) 0%,#8B5CF6 100%)', boxShadow: '0 0 0 1px rgba(99,102,241,.3),0 6px 20px rgba(99,102,241,.3)' }}>🎯</div>
            <span style={{ fontFamily: 'var(--font-display)', fontSize: 22, fontWeight: 800, background: 'linear-gradient(120deg,#fff 40%,var(--accent-light))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>hireable</span>
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--text3)', letterSpacing: 0.2 }}>AI Resume Intelligence — Private Beta</div>
        </div>

        <div style={{ background: 'var(--surface)', border: '1px solid var(--border2)', borderRadius: 14, padding: 28, boxShadow: 'var(--shadow-lg)' }}>
          {/* Tabs */}
          <div style={{ display: 'flex', background: 'var(--surface2)', borderRadius: 8, padding: 3, marginBottom: 22, gap: 3 }}>
            {(['login', 'register'] as const).map(t => (
              <button key={t} onClick={() => setTab(t)} style={{ flex: 1, padding: '7px', fontSize: 12.5, fontWeight: 600, border: 'none', cursor: 'pointer', borderRadius: 6, fontFamily: 'var(--font-body)', transition: 'all .15s', background: tab === t ? 'var(--accent)' : 'transparent', color: tab === t ? '#fff' : 'var(--text2)' }}>
                {t === 'login' ? 'Sign In' : 'Create Account'}
              </button>
            ))}
          </div>

          {error && <div style={{ fontSize: 12, padding: '9px 12px', borderRadius: 7, marginBottom: 16, background: 'var(--red-bg)', color: 'var(--red)', border: '1px solid var(--red-border)' }}>{error}</div>}

          <div style={{ marginBottom: 14 }}>
            <label style={{ fontSize: 11, fontWeight: 700, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: .7, display: 'block', marginBottom: 6 }}>Email</label>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@company.com" autoComplete="email" onKeyDown={e => e.key === 'Enter' && submit()} style={inputStyle}
              onFocus={e => { e.target.style.borderColor = 'var(--accent)'; e.target.style.boxShadow = '0 0 0 3px var(--accent-glow)' }}
              onBlur={e => { e.target.style.borderColor = 'var(--border2)'; e.target.style.boxShadow = 'none' }} />
          </div>
          <div style={{ marginBottom: 22 }}>
            <label style={{ fontSize: 11, fontWeight: 700, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: .7, display: 'block', marginBottom: 6 }}>Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="••••••••" autoComplete="current-password" onKeyDown={e => e.key === 'Enter' && submit()} style={inputStyle}
              onFocus={e => { e.target.style.borderColor = 'var(--accent)'; e.target.style.boxShadow = '0 0 0 3px var(--accent-glow)' }}
              onBlur={e => { e.target.style.borderColor = 'var(--border2)'; e.target.style.boxShadow = 'none' }} />
          </div>
          <button onClick={submit} disabled={loading} style={{ width: '100%', padding: '10px', borderRadius: 7, border: 'none', background: 'var(--accent)', color: '#fff', fontSize: 13.5, fontWeight: 700, cursor: loading ? 'not-allowed' : 'pointer', fontFamily: 'var(--font-body)', opacity: loading ? 0.6 : 1, boxShadow: '0 1px 2px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.1)', transition: 'all .12s' }}>
            {loading ? 'Please wait…' : tab === 'login' ? 'Sign In' : 'Create Account'}
          </button>
        </div>
      </div>
    </div>
  )
}
