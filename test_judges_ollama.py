#!/usr/bin/env python3
"""
Test script to validate judge prompts using Ollama
"""

import asyncio
import json
import os
from datetime import datetime
import sys
sys.path.append('.')

# Ollama integration
import aiohttp

async def call_ollama(prompt, model="llama3.1", temperature=0.1):
    """Call Ollama API"""
    try:
        url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 2000
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"success": True, "data": result.get("response", "")}
                else:
                    return {"success": False, "error": f"HTTP {response.status}"}
                    
    except Exception as e:
        return {"success": False, "error": str(e)}

# Load sample data
def load_sample_data():
    """Load one resume and one job for testing"""
    import pandas as pd
    import re
    
    # Load resumes
    resumes_df = pd.read_csv('data/resumes.csv')
    resume = resumes_df.iloc[0]  # Get first resume
    
    # Load jobs  
    jobs_df = pd.read_csv('data/jobs.csv')
    job = jobs_df.iloc[0]  # Get first job
    
    # Clean job description - remove HTML/JavaScript
    job_desc = job['description']
    # Remove HTML tags and scripts
    job_desc = re.sub(r'<script[^>]*>.*?</script>', '', job_desc, flags=re.DOTALL | re.IGNORECASE)
    job_desc = re.sub(r'<[^>]+>', '', job_desc)
    job_desc = re.sub(r'\.js[^>]*>', '', job_desc)
    job_desc = re.sub(r'window\.[^;]*;', '', job_desc)
    job_desc = re.sub(r'\(function[^)]*\)', '', job_desc)
    # Clean up extra whitespace
    job_desc = re.sub(r'\s+', ' ', job_desc).strip()
    
    return {
        'resume_text': resume['text'],
        'resume_name': resume['name'],
        'job_description': job_desc,
        'job_title': job['title'],
        'company': job['company']
    }

async def test_judge_prompt(data, judge_name, system_prompt, user_prompt_template, model):
    """Generic test function for any judge"""
    print(f"\n" + "="*60)
    print(f"🧪 TESTING {judge_name.upper()} JUDGE")
    print(f"🤖 Model: {model}")
    print("="*60)
    
    start_time = datetime.now()
    
    try:
        # Format the user prompt
        user_prompt = user_prompt_template.format(
            job_description=data['job_description'],
            resume_text=data['resume_text']
        )
        
        # Combine prompts
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        print(f"📝 Prompt length: {len(combined_prompt)} characters")
        
        # Call Ollama
        result = await call_ollama(combined_prompt, model, temperature=0.1)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"✅ Success: {result.get('success', False)}")
        
        if result.get('success'):
            response_text = result['data']
            
            # Try to parse JSON
            try:
                # Clean response - Ollama might add extra text
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].rsplit('```', 1)[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].rsplit('```', 1)[0].strip()
                
                # Find JSON object in response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
                
                judge_data = json.loads(response_text)
                
                print(f"✅ JSON parsed successfully")
                
                # Calculate score based on judge type
                score = 0.0
                if judge_name == 'skills':
                    skill_analysis = judge_data.get('skill_analysis', [])
                    if skill_analysis:
                        total_skills = len(skill_analysis)
                        present_skills = sum(1 for skill in skill_analysis if skill.get('is_present') == 1)
                        score = present_skills / total_skills if total_skills > 0 else 0.0
                        
                        print(f"📊 Skills analyzed: {total_skills}")
                        print(f"🎯 Skills Coverage: {score*100:.1f}%")
                        print(f"✅ Present Skills: {present_skills}")
                        print(f"❌ Missing Skills: {total_skills - present_skills}")
                        
                        # Show first few skills
                        print("\n📋 Sample Skills Analysis:")
                        for i, skill in enumerate(skill_analysis[:2]):
                            status = "✅" if skill.get('is_present') == 1 else "❌"
                            print(f"   {status} {skill.get('skill_requirement', 'N/A')[:60]}...")
                            print(f"      Reasoning: {skill.get('reasoning', 'N/A')[:60]}...")
                
                elif judge_name == 'experience':
                    resp_analysis = judge_data.get('responsibility_analysis', [])
                    if resp_analysis:
                        total_resp = len(resp_analysis)
                        score_sum = sum(resp.get('experience_score', 0) for resp in resp_analysis)
                        score = score_sum / total_resp if total_resp > 0 else 0.0
                        
                        print(f"📊 Responsibilities analyzed: {total_resp}")
                        print(f"🎯 Experience Depth: {score*100:.1f}%")
                        
                        # Show first few responsibilities
                        print("\n📋 Sample Experience Analysis:")
                        for i, resp in enumerate(resp_analysis[:2]):
                            score_val = resp.get('experience_score', 0)
                            if score_val == 1.0:
                                status = "✅"
                                label = "Exceeds/Meets"
                            elif score_val == 0.5:
                                status = "⚠️"
                                label = "Partial"
                            else:
                                status = "❌"
                                label = "Unmet"
                            print(f"   {status} {resp.get('responsibility', 'N/A')[:60]}... - {label}")
                            print(f"      Reasoning: {resp.get('reasoning', 'N/A')[:60]}...")
                
                print(f"\n📝 Overall Reasoning: {judge_data.get('overall_reasoning', 'N/A')[:150]}...")
                
                return {
                    'judge': judge_name,
                    'model': model,
                    'success': True,
                    'duration': duration,
                    'score': score,
                    'data': judge_data
                }
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON Parse Error: {e}")
                print(f"📄 Raw Response: {response_text[:500]}...")
                return {
                    'judge': judge_name,
                    'model': model,
                    'success': False,
                    'duration': duration,
                    'error': f'JSON Parse Error: {e}'
                }
        else:
            print(f"❌ Ollama Error: {result}")
            return {
                'judge': judge_name,
                'model': model,
                'success': False,
                'duration': duration,
                'error': str(result)
            }
            
    except Exception as e:
        print(f"💥 Exception: {str(e)}")
        return {
            'judge': judge_name,
            'model': model,
            'success': False,
            'duration': 0,
            'error': str(e)
        }

async def main():
    """Run judge prompt tests with Ollama"""
    print("🚀 Starting Judge Prompt Validation Tests with Ollama")
    print(f"📅 Test Time: {datetime.now().isoformat()}")
    
    # Check if Ollama is running
    try:
        test_result = await call_ollama("Hello", "llama3.1:8b")
        if not test_result.get('success'):
            print("❌ Error: Ollama is not running or not accessible")
            print("   Please make sure Ollama is installed and running:")
            print("   - Install: https://ollama.ai/")
            print("   - Start: ollama serve")
            print("   - Pull model: ollama pull llama3.1:8b")
            return
        else:
            print("✅ Ollama is accessible")
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return
    
    # Load sample data
    print("\n📂 Loading sample data...")
    try:
        data = load_sample_data()
        print(f"✅ Resume: {data['resume_name']}")
        print(f"✅ Job: {data['job_title']} at {data['company']}")
        print(f"📄 Resume length: {len(data['resume_text'])} chars")
        print(f"📄 Job description length: {len(data['job_description'])} chars")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Judge prompts
    skills_system = """You are a ruthless technical recruiter evaluating a candidate's Resume against a Job Description.
Your PRIMARY job is to analyze skills coverage using statement-by-statement evaluation.

MANDATORY EVALUATION ALGORITHM:
STEP 1: Extract all explicitly required technical skills, soft skills, and qualifications from the Job Description.
STEP 2: For EACH required skill, analyze if the candidate has demonstrated this skill in their resume.
STEP 3: Break down your analysis into individual statements about each skill requirement.
STEP 4: For each statement, determine if the skill is present (binary 1) or missing (binary 0).
STEP 5: Provide brief reasoning for each binary classification.

CRITICAL: Do not calculate final scores. Output strict JSON only with statement-by-statement analysis."""

    skills_user = """Job Description:
{job_description}

Candidate Resume:
{resume_text}

Perform the skills coverage analysis using statement-by-statement evaluation and output the JSON.

OUTPUT FORMAT:
{{
    "skill_analysis": [
        {{
            "skill_requirement": "<exact skill requirement from JD>",
            "is_present": <1 or 0>,
            "reasoning": "<brief explanation of why this skill is present (1) or missing (0)>",
            "resume_evidence": "<exact quote from resume if skill is present, or 'Not found' if missing>"
        }}
    ],
    "overall_reasoning": "<final summary of how the candidate's skills align with job requirements>"
}}

Note: is_present MUST be exactly 1 (skill found) or 0 (skill missing)."""

    exp_system = """You are an expert hiring manager conducting a deep-dive relevancy evaluation.
Your job is to determine if the candidate's experience depth matches the Job Description's scope using statement-by-statement analysis.

MANDATORY REASONING ALGORITHM:
STEP 1: Identify the 3 most critical core responsibilities in the Job Description.
STEP 2: For EACH core responsibility, analyze if the candidate has performed work of similar scope, complexity, and scale.
STEP 3: Break down your analysis into individual statements about each responsibility.
STEP 4: For each statement, assign a binary classification: 1 (exceeds_or_meets), 0.5 (partial_match), or 0 (unmet).
STEP 5: Provide brief reasoning for each classification.

CRITICAL: Do not calculate final scores. Output strict JSON only with statement-by-statement analysis."""

    exp_user = """Job Description:
{job_description}

Candidate Resume:
{resume_text}

Perform the experience depth analysis using statement-by-statement evaluation and output the JSON.

OUTPUT FORMAT:
{{
    "responsibility_analysis": [
        {{
            "responsibility": "<exact responsibility requirement from JD>",
            "experience_score": <1.0, 0.5, or 0.0>,
            "reasoning": "<brief explanation of why this score was assigned>",
            "resume_evidence": "<exact quote from resume showing relevant experience>"
        }}
    ],
    "overall_reasoning": "<final summary of how the candidate's experience aligns with job responsibilities>"
}}

Note: experience_score MUST be exactly 1.0 (exceeds_or_meets), 0.5 (partial_match), or 0.0 (unmet)."""
    
    # Test models - Best for Complex Analysis (2026 benchmarks)
    models = [
        "deepseek-r1:14b",     # Best for chain-of-thought logic & reasoning
        "phi4:14b",            # Best for STEM & structured logic  
        "llama3.1:8b",         # Good all-rounder (currently pulling)
        "qwen2.5-coder:7b"     # Best for 8GB GPU laptops
    ]
    
    results = []
    
    for model in models:
        print(f"\n🔄 Testing with model: {model}")
        
        # Test Skills Judge
        skills_result = await test_judge_prompt(data, 'skills', skills_system, skills_user, model)
        results.append(skills_result)
        
        # Test Experience Judge  
        exp_result = await test_judge_prompt(data, 'experience', exp_system, exp_user, model)
        results.append(exp_result)
        
        # Small delay between models
        await asyncio.sleep(1)
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    total_duration = sum(r['duration'] for r in results)
    successful = sum(1 for r in results if r['success'])
    
    print(f"⏱️  Total Duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print(f"✅ Successful Tests: {successful}/{len(results)}")
    print(f"📈 Success Rate: {successful/len(results)*100:.1f}%")
    
    print("\n📋 Individual Results:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        judge_name = result['judge'].upper()
        model_short = result['model']
        duration = result['duration']
        score = result.get('score', 'N/A')
        if isinstance(score, float):
            score = f"{score*100:.1f}%"
        print(f"   {status} {judge_name} ({model_short}): {duration:.1f}s | Score: {score}")
    
    # Save detailed results
    output_file = f"test_results_ollama_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'test_time': datetime.now().isoformat(),
            'sample_data': {
                'resume_name': data['resume_name'],
                'job_title': data['job_title'],
                'company': data['company'],
                'resume_length': len(data['resume_text']),
                'job_length': len(data['job_description'])
            },
            'models_tested': models,
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
