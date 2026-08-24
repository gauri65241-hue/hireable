import { useEffect, useState, useCallback } from 'react'
import { useStore, type Resume, type Job } from '@/store'
import { useAuth } from '@/hooks/useAuth'
import { api } from '@/lib/api'
import { Layout } from '@/components/layout/Layout'
import { AuthOverlay } from '@/components/auth/AuthOverlay'
import { Health } from '@/pages/Health'
import { Resumes } from '@/pages/Resumes'
import { Jobs } from '@/pages/Jobs'
import { Analyzer } from '@/pages/Analyzer'
import { Results } from '@/pages/Results'
import { AtsChecker } from '@/pages/AtsChecker'
import { CareerPaths } from '@/pages/CareerPaths'
import { Interview } from '@/pages/Interview'
import { Benchmark } from '@/pages/Benchmark'

export default function App() {
  const { currentPage, setResumes, setJobs, theme } = useStore()

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])
  const { isAuthenticated } = useAuth()
  const [resumeCount, setResumeCount] = useState(0)
  const [jobCount, setJobCount] = useState(0)

  const fetchData = useCallback(async () => {
    if (!isAuthenticated) return
    try {
      const [r, j] = await Promise.all([
        api<Resume[]>('/resumes'),
        api<Job[]>('/jobs'),
      ])
      const rm = Object.fromEntries(r.map(x => [String(x.resume_id), x]))
      const jm = Object.fromEntries(j.map(x => [String(x.job_id), x]))
      setResumes(rm)
      setJobs(jm)
      setResumeCount(r.length)
      setJobCount(j.length)
    } catch (e) { console.error('fetch failed', e) }
  }, [isAuthenticated, setResumes, setJobs])

  useEffect(() => { void fetchData() }, [fetchData])

  const pages: Record<string, React.ReactNode> = {
    health:         <Health resumeCount={resumeCount} jobCount={jobCount} />,
    resumes:        <Resumes />,
    jobs:           <Jobs />,
    analyzer:       <Analyzer />,
    results:        <Results />,
    'ats-checker':  <AtsChecker />,
    'career-paths': <CareerPaths />,
    interview:      <Interview />,
    benchmark:      <Benchmark />,
  }

  return (
    <>
      <AuthOverlay visible={!isAuthenticated} />
      {isAuthenticated && (
        <Layout onRefresh={fetchData}>
          {pages[currentPage] ?? pages['health']}
        </Layout>
      )}
    </>
  )
}
