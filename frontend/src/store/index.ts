import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface Resume {
  resume_id: number
  name: string
  title: string
  years_of_experience: number
  skills: string
  text: string
  created_at: string
  user_id: number
}

export interface Job {
  job_id: number
  company: string
  title: string
  experience_required: string
  skills_required: string
  description: string
  source_url?: string
  applied?: boolean
  created_at: string
  user_id: number
}

interface AppStore {
  token: string | null
  setToken: (t: string | null) => void
  clearToken: () => void

  resumes: Record<string, Resume>
  jobs: Record<string, Job>
  setResumes: (r: Record<string, Resume>) => void
  setJobs: (j: Record<string, Job>) => void

  selectedResumes: Set<string>
  selectedJobs: Set<string>
  toggleResume: (id: string) => void
  toggleJob: (id: string) => void
  clearSelections: () => void

  currentPage: string
  setCurrentPage: (p: string) => void

  theme: 'dark' | 'light'
  toggleTheme: () => void
}

export const useStore = create<AppStore>()(
  persist(
    (set, get) => ({
      token: null,
      setToken: (t) => set({ token: t }),
      clearToken: () => set({ token: null }),

      resumes: {},
      jobs: {},
      setResumes: (r) => set({ resumes: r }),
      setJobs: (j) => set({ jobs: j }),

      selectedResumes: new Set(),
      selectedJobs: new Set(),
      toggleResume: (id) => {
        const s = new Set(get().selectedResumes)
        s.has(id) ? s.delete(id) : s.add(id)
        set({ selectedResumes: s })
      },
      toggleJob: (id) => {
        const s = new Set(get().selectedJobs)
        s.has(id) ? s.delete(id) : s.add(id)
        set({ selectedJobs: s })
      },
      clearSelections: () => set({ selectedResumes: new Set(), selectedJobs: new Set() }),

      currentPage: 'health',
      setCurrentPage: (p) => set({ currentPage: p }),

      theme: 'light',
      toggleTheme: () => set((s) => ({ theme: s.theme === 'dark' ? 'light' : 'dark' })),
    }),
    {
      name: 'hireable-store',
      partialize: (s) => ({ token: s.token, theme: s.theme }),
    }
  )
)
