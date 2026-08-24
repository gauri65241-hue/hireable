import { useEffect, useState } from 'react'
import { useStore } from '@/store'
import { isTokenExpired } from '@/lib/api'

export function useAuth() {
  const { token, setToken, clearToken } = useStore()
  const [isAuthenticated, setIsAuthenticated] = useState(false)

  useEffect(() => {
    if (token && !isTokenExpired(token)) {
      setIsAuthenticated(true)
    } else {
      clearToken()
      setIsAuthenticated(false)
    }
  }, [token, clearToken])

  const login = (newToken: string) => {
    setToken(newToken)
    setIsAuthenticated(true)
  }

  const logout = () => {
    clearToken()
    setIsAuthenticated(false)
  }

  const getEmail = (): string => {
    if (!token) return ''
    try { return (JSON.parse(atob(token.split('.')[1])) as { email?: string }).email ?? '' }
    catch { return '' }
  }

  return { isAuthenticated, token, login, logout, getEmail }
}
