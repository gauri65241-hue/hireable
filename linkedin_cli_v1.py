#!/usr/bin/env python3
"""
hireable CLI - Data Loading and Processing Functions
"""

import pandas as pd
import json
import os
from typing import List, Dict, Any

def load_resumes(csv_path: str) -> List[Dict[str, Any]]:
    """Load resumes from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        resumes = []
        
        for _, row in df.iterrows():
            resume = {
                'resume_id': str(row.get('resume_id', '')),
                'name': str(row.get('name', '')),
                'title': str(row.get('title', '')),
                'experience_years': str(row.get('experience_years', '')),
                'skills': str(row.get('skills', '')),
                'text': str(row.get('text', ''))
            }
            resumes.append(resume)
        
        print(f"Successfully loaded {len(resumes)} resumes")
        return resumes
        
    except Exception as e:
        print(f"Error loading resumes: {e}")
        return []

def load_jobs(csv_path: str) -> List[Dict[str, Any]]:
    """Load jobs from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        jobs = []
        
        for _, row in df.iterrows():
            job = {
                'job_id': str(row.get('job_id', '')),
                'title': str(row.get('title', '')),
                'company': str(row.get('company', '')),
                'experience_required': str(row.get('experience_required', '')),
                'skills_required': str(row.get('skills_required', '')),
                'description': str(row.get('description', '')),
                'source_url': str(row.get('source_url', '')),
                'applied': row.get('applied', '0') == '1',
            }
            jobs.append(job)
        
        print(f"Successfully loaded {len(jobs)} jobs")
        return jobs
        
    except Exception as e:
        print(f"Error loading jobs: {e}")
        return []

def load_skills_taxonomy(json_path: str) -> Dict[str, Any]:
    """Load skills taxonomy from JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            taxonomy = json.load(f)
        print(f"Successfully loaded skills taxonomy from {json_path}")
        return taxonomy
    except FileNotFoundError:
        print(f"Skills taxonomy file not found: {json_path}")
        return {}
    except Exception as e:
        print(f"Error loading skills taxonomy: {e}")
        return {}

def validate_data(resumes: List[Dict], jobs: List[Dict]) -> bool:
    """Validate that required fields are present"""
    required_resume_fields = ['resume_id', 'name', 'text']
    required_job_fields = ['job_id', 'title', 'description']
    
    # Check resumes
    for resume in resumes:
        for field in required_resume_fields:
            if not resume.get(field):
                print(f"Missing required field '{field}' in resume {resume.get('resume_id', 'unknown')}")
                return False
    
    # Check jobs
    for job in jobs:
        for field in required_job_fields:
            if not job.get(field):
                print(f"Missing required field '{field}' in job {job.get('job_id', 'unknown')}")
                return False
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    # Test data loading
    print("Testing data loading functions...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resumes_csv = os.path.join(base_dir, "data", "resumes.csv")
    jobs_csv = os.path.join(base_dir, "data", "jobs.csv")
    skills_json = os.path.join(base_dir, "data", "skills_taxonomy.json")
    
    resumes = load_resumes(resumes_csv)
    jobs = load_jobs(jobs_csv)
    taxonomy = load_skills_taxonomy(skills_json)
    
    if validate_data(resumes, jobs):
        print("✅ All data loaded and validated successfully!")
    else:
        print("❌ Data validation failed")
