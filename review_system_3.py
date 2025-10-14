import os
import sys
import json
import re
import datetime
import PyPDF2
import statistics
import time
from typing import List, Dict, Any, Tuple, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    import streamlit as st
except ImportError:
    st = None

# Initialize OpenAI API key
print("Environment variables:", os.environ)
openai_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI API key from env: {'Found' if openai_key else 'Not found'}")

if not openai_key and st:
    try:
        print("Checking Streamlit secrets...")
        if 'openai' in st.secrets and 'api_key' in st.secrets.openai:
            openai_key = st.secrets.openai.api_key
            os.environ["OPENAI_API_KEY"] = openai_key
            print("Found API key in Streamlit secrets")
    except Exception as e:
        print(f"Error accessing Streamlit secrets: {e}")

if not openai_key:
    raise EnvironmentError(
        "OpenAI API key not found. Please set it in Streamlit secrets or as an environment variable."
    )

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Constants
MAX_RETRIES = 3
MAX_WORKERS = 4  # Adjust based on your rate limits
CHUNK_SIZE = 2000

# Initialize LLM once
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    request_timeout=60,
    max_retries=2
)

# Define prompt template once
ANALYSIS_PROMPT_TEMPLATE = """
You are an expert reviewer. Your ONLY job is to check if the mentor followed the official Analytics Vidhya mentorship guidelines.
Never judge technical/subject knowledge. ONLY comment on adherence to guidelines, professionalism, session structure, and communication.

CRITICAL: Score ONLY based on guideline adherence. Do NOT score based on technical knowledge, subject expertise, or correctness of answers.

CONVERSATION METRICS:
{metrics}

CHUNK {chunk_number} CONTENT:
{chunk}

OFFICIAL GUIDELINES (summary):
{guidelines}

Please fill the following JSON object (skip technical points):

{{
  "summary": "One sentence on mentor's compliance to guidelines.",
  "positive_guideline_behaviors": [
    {{"guideline": "Which guideline point was followed", "example": "Example phrase or behavior"}}
  ],
  "guideline_violations": [
    {{
      "guideline": "Which guideline was NOT followed",
      "severity": <int 1-10>,
      "evidence": "Exact phrase/example from chunk",
      "impact": "Why this matters for guideline compliance"
    }}
  ],
  "improvement_suggestions": [
    {{
      "title": "[Specific guideline area for improvement]",
      "suggestion": "[Actionable recommendation on following guidelines better. Use second person 'you' and be specific. Keep it professional and supportive.]"
    }}
  ],
  "guideline_professionalism_score": <int 1-10, based ONLY on following professionalism guidelines>,
  "guideline_session_flow_score": <int 1-10, based ONLY on following session structure guidelines>,
  "overall_guideline_compliance": <int 1-10, based ONLY on overall adherence to all guidelines>
}}
"""

# New prompt template for overall session summary
SESSION_SUMMARY_PROMPT = """
You are an expert analyst reviewing a mentorship session transcript. Provide a comprehensive overview of the entire session.

FULL TRANSCRIPT:
{transcript}

Please provide a detailed summary that covers:
1. Main topics discussed during the session
2. Key learning outcomes for the student
3. Overall quality and effectiveness of the mentorship
4. Student's engagement and participation level
5. Mentor's teaching approach and methodology

Write a cohesive paragraph (150-250 words) that captures the essence of this mentorship session.
"""

# New prompt template for session checklist
SESSION_CHECKLIST_PROMPT = """
You are an expert reviewer analyzing a mentorship session transcript. Answer the following questions based on evidence from the transcript.

FULL TRANSCRIPT:
{transcript}

For each question, provide:
1. A clear YES or NO answer
2. A brief explanation (3-4 sentences) with specific evidence from the transcript

Return your response as a JSON object with this exact structure:

{{
  "checklist": [
    {{
      "question": "Did mentor have camera feed with Virtual Background or Blur background?",
      "answer": "YES/NO",
      "explanation": "Brief explanation with evidence"
    }},
    {{
      "question": "Was there any network issues or background noise?",
      "answer": "YES/NO",
      "explanation": "Brief explanation with evidence"
    }},
    {{
      "question": "Did mentor login on Time?",
      "answer": "YES/NO/UNCLEAR",
      "explanation": "Brief explanation with evidence"
    }},
    {{
      "question": "Did mentor look like he/she knows the student's profile or have asked the same from student?",
      "answer": "YES/NO",
      "explanation": "Brief explanation with evidence"
    }},
    {{
      "question": "Did mentor ask students what Challenges they are facing currently?",
      "answer": "YES/NO",
      "explanation": "Brief explanation with evidence"
    }},
    {{
      "question": "Identify the specific issue/concern by discussing with student (If Technical Session)",
      "answer": "YES/NO/N/A",
      "explanation": "Brief explanation with evidence"
    }},
    {{
      "question": "Commitment taken from student (On their learning time, weekly review of their own progress)",
      "answer": "YES/NO",
      "explanation": "Brief explanation with evidence"
    }},
    {{
      "question": "Did the mentor summarize the session, set expectations, and take clear commitments from the student on specific milestones?",
      "answer": "YES/NO",
      "explanation": "Brief explanation with evidence"
    }},
    {{
      "question": "Did mentor said anything negative about AV or its courses / Has mentor shared some personal details?",
      "answer": "YES/NO",
      "explanation": "Brief explanation with evidence"
    }}
  ]
}}
"""


@lru_cache(maxsize=32)
def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF with caching to avoid repeated reads."""
    try:
        if not os.path.exists(pdf_path):
            return "Standard mentorship guidelines apply."
            
        # Check if the file has been modified since last read
        mtime = os.path.getmtime(pdf_path)
        cache_key = f"{pdf_path}:{mtime}"
        
        if hasattr(extract_pdf_text, '_cache') and cache_key in extract_pdf_text._cache:
            return extract_pdf_text._cache[cache_key]
            
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "\n".join(
                page.extract_text() 
                for page in pdf_reader.pages 
                if page.extract_text()
            )
            
            # Cache the result
            if not hasattr(extract_pdf_text, '_cache'):
                extract_pdf_text._cache = {}
            extract_pdf_text._cache[cache_key] = text
            
            return text
    except Exception as e:
        if st:
            st.warning(f"Error reading PDF: {str(e)}")
        return "Standard mentorship guidelines apply."


def extract_json_from_text(text: str) -> str:
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text


def chunk_text_intelligently(text: str, max_chars: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks, trying to keep conversations intact."""
    if not text:
        return []
        
    lines = text.split('\n')
    if not lines:
        return []
        
    chunks = []
    current_chunk = []
    current_length = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        line_len = len(line) + 1  # +1 for newline
        
        # Check if we need to start a new chunk
        if (current_length + line_len > max_chars and current_chunk) or \
           (line.lower().startswith('mentor:') and current_length > max_chars * 0.6):
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_length = 0
            
        current_chunk.append(line)
        current_length += line_len
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks


def analyze_conversation_metrics(transcript_text):
    lines = transcript_text.split('\n')
    mentor_lines = [line for line in lines if line.strip().lower().startswith('mentor:')]
    student_lines = [line for line in lines if line.strip().lower().startswith('student:')]
    mentor_words = sum(len(line.split()) for line in mentor_lines)
    student_words = sum(len(line.split()) for line in student_lines)
    total_words = mentor_words + student_words
    return {
        "mentor_talk_ratio": mentor_words / total_words if total_words else 0.5,
        "student_talk_ratio": student_words / total_words if total_words else 0.5,
        "total_exchanges": len(mentor_lines) + len(student_lines)
    }


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry_error_callback=lambda _: None
)
def analyze_single_chunk(args: Tuple[int, str, str, dict]) -> Optional[dict]:
    """Process a single chunk of text with retry logic."""
    chunk_idx, chunk, guidelines, conversation_metrics = args
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _process_chunk():
        try:
            prompt = ANALYSIS_PROMPT_TEMPLATE.format(
                chunk=chunk,
                guidelines=guidelines,
                metrics=json.dumps(conversation_metrics, indent=2),
                chunk_number=chunk_idx + 1
            )
            
            response = llm.invoke(prompt)
            analysis_text = response.content if hasattr(response, 'content') else str(response)
            json_text = extract_json_from_text(analysis_text)
            analysis_data = json.loads(json_text)
            
            return {
                "chunk_index": chunk_idx,
                "analysis": analysis_data
            }
        except json.JSONDecodeError as e:
            print(f"JSON decode error in chunk {chunk_idx}: {e}")
            print(f"Response was: {analysis_text}")
            raise
        except Exception as e:
            print(f"Error in chunk {chunk_idx}: {str(e)}")
            raise
    
    try:
        return _process_chunk()
    except Exception as e:
        print(f"Failed to process chunk {chunk_idx} after {MAX_RETRIES} attempts")
        return {
            "chunk_index": chunk_idx,
            "analysis": {
                "summary": f"Chunk {chunk_idx + 1} analysis failed",
                "positive_guideline_behaviors": [],
                "guideline_violations": [{
                    "guideline": "Analysis Error", 
                    "severity": 8, 
                    "evidence": str(type(e).__name__), 
                    "impact": f"Analysis failed: {str(e)[:200]}"
                }],
                "improvement_suggestions": [{
                    "title": "Review Error",
                    "suggestion": "Automated review failed for this segment."
                }],
                "guideline_professionalism_score": 5,
                "guideline_session_flow_score": 5,
                "overall_guideline_compliance": 5
            }
        }

def _analyze_chunk_wrapper(args):
    """Wrapper function to process a single chunk of text."""
    return analyze_single_chunk(args)

def deep_analyze_chunks(chunks: List[str], guidelines: str, conversation_metrics: dict) -> List[dict]:
    """Process multiple chunks in parallel with rate limiting."""
    results = []
    
    # Prepare arguments for each chunk
    args_list = [(i, chunk, guidelines, conversation_metrics) 
                for i, chunk in enumerate(chunks)]
    
    # Process chunks in parallel with rate limiting
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_chunk = {
            executor.submit(_analyze_chunk_wrapper, args): args 
            for args in args_list
        }
        
        # Process results as they complete
        for future in as_completed(future_to_chunk):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing chunk: {e}")
    
    # Sort results by chunk index to maintain original order
    results.sort(key=lambda x: x.get("chunk_index", 0))
    return results


def aggregate_guideline_assessment(chunk_analyses):
    """Aggregate scores and feedback from chunk analyses based ONLY on guideline adherence.
    Scoring focuses exclusively on how well the mentor followed the official guidelines,
    not on technical knowledge or subject expertise.
    """
    import statistics

    # Trackers for guideline-specific scores
    professionalism_scores, session_flow_scores, compliance_scores = [], [], []
    all_positive_behaviors, all_violations, all_improvements = [], [], []

    for ca in (chunk_analyses or []):
        analysis = (ca or {}).get("analysis", {})

        # Extract guideline-based scores only
        prof_score = analysis.get("guideline_professionalism_score")
        flow_score = analysis.get("guideline_session_flow_score")
        compliance_score = analysis.get("overall_guideline_compliance")
        
        if isinstance(prof_score, (int, float)):
            professionalism_scores.append(float(prof_score))
        if isinstance(flow_score, (int, float)):
            session_flow_scores.append(float(flow_score))
        if isinstance(compliance_score, (int, float)):
            compliance_scores.append(float(compliance_score))

        all_positive_behaviors.extend(analysis.get("positive_guideline_behaviors", []) or [])
        all_violations.extend(analysis.get("guideline_violations", []) or [])
        
        for imp in analysis.get("improvement_suggestions", []) or []:
            if isinstance(imp, str):
                imp = {"title": imp, "suggestion": imp}
            if imp and imp not in all_improvements:
                all_improvements.append(imp)

    n_chunks = max(1, len(chunk_analyses or []))

    def _sev(item, default=5.0):
        try:
            return float(item.get("severity", default))
        except Exception:
            return float(default)

    # Calculate violation impact
    severe_violations = [v for v in all_violations if _sev(v) >= 8]
    moderate_violations = [v for v in all_violations if 5 <= _sev(v) < 8]
    minor_violations = [v for v in all_violations if _sev(v) < 5]

    # Normalize by chunks
    pos_rate = len(all_positive_behaviors) / n_chunks
    sev_rate = len(severe_violations) / n_chunks
    mod_rate = len(moderate_violations) / n_chunks
    minor_rate = len(minor_violations) / n_chunks

    # Heuristic based on guideline adherence
    def _guideline_heuristic():
        # Base score starts at 7.0 (neutral guideline following)
        base = 7.0
        # Reward positive guideline behaviors
        base += min(2.0, 0.8 * pos_rate)
        # Penalize violations based on severity
        base -= min(3.0, 1.2 * sev_rate + 0.6 * mod_rate + 0.2 * minor_rate)
        return max(1.0, min(10.0, base))

    def _robust_mean(scores):
        if not scores:
            return _guideline_heuristic()
        m = statistics.mean(scores)
        sd = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        # If all scores are flat/default, use heuristic instead
        if abs(m - 5.0) < 0.2 and sd < 0.2:
            return _guideline_heuristic()
        return max(1.0, min(10.0, m))

    # Calculate average scores for each guideline dimension
    avg_professionalism = _robust_mean(professionalism_scores)
    avg_session_flow = _robust_mean(session_flow_scores)
    avg_compliance = _robust_mean(compliance_scores)

    # Weighted score based purely on guideline adherence
    # Equal weight to all guideline aspects
    weights = {
        "professionalism": 0.35,
        "session_flow": 0.35,
        "compliance": 0.30,
    }

    base_score = (
        avg_professionalism * weights["professionalism"] +
        avg_session_flow * weights["session_flow"] +
        avg_compliance * weights["compliance"]
    )

    # Apply violation penalties
    violation_penalty = 0
    if severe_violations:
        violation_penalty += min(2.0, 0.8 * sev_rate)
    if moderate_violations:
        violation_penalty += min(1.5, 0.5 * mod_rate)
    if minor_violations:
        violation_penalty += min(0.8, 0.2 * minor_rate)

    # Apply positive behavior bonus
    positive_bonus = min(1.5, 0.6 * pos_rate)

    # Calculate final score (1-10 scale)
    raw_score = base_score + positive_bonus - violation_penalty
    raw_score = max(1.0, min(10.0, raw_score))

    # Calibration: Center around 7.5 for sessions with reasonable guideline adherence
    neutral_point = 7.0
    scale = 1.05
    calibrated = 7.5 + (raw_score - neutral_point) * scale
    final_score = max(1.0, min(10.0, calibrated))

    # Convert to 70-100 scale (previously 0-100)
    scaled_score = round(70 + (final_score * 3.0), 1)  # Maps 0-10 to 70-100
    scaled_score = max(70.0, min(100.0, scaled_score))  # Ensure within 70-100 range

    # Scale detailed scores to 70-100 as well
    def scale_detail_score(score):
        return round(70 + (score * 3.0), 1)

    return {
        "overall_score": scaled_score,
        "detailed_scores": {
            "guideline_professionalism": scale_detail_score(avg_professionalism),
            "guideline_session_flow": scale_detail_score(avg_session_flow),
            "guideline_compliance": scale_detail_score(avg_compliance),
        },
        "professionalism": scale_detail_score(avg_professionalism),
        "session_flow": scale_detail_score(avg_session_flow),
        "overall_guideline_compliance": scale_detail_score(avg_compliance),
        "positive_behaviors": all_positive_behaviors,
        "violations": all_violations,
        "improvements": all_improvements,
        "violation_summary": {
            "severe": len(severe_violations),
            "moderate": len(moderate_violations),
            "minor": len(minor_violations)
        }
    }

def extract_mentor_name(transcript_data, transcript_text):
    exclude_names = ['student', 'unknown', '']
    names = [item.get('speaker_name','').strip() for item in transcript_data if isinstance(item, dict)]
    mentor_candidates = [n for n in set(names) if n.lower() not in exclude_names]
    if mentor_candidates:
        freq = {name: names.count(name) for name in mentor_candidates}
        return max(freq.items(), key=lambda x: x[1])[0]
    lines = transcript_text.strip().split('\n')[:20]
    candidates = {}
    for line in lines:
        if ':' in line:
            speaker = line.split(':', 1)[0].strip()
            if speaker.lower() not in exclude_names and 1 < len(speaker.split()) <= 3:
                candidates[speaker] = candidates.get(speaker, 0) + 1
    if candidates:
        return max(candidates.items(), key=lambda x: x[1])[0]
    return "Mentor"


def guideline_feedback_email(mentor_name, assessment):
    fixed_strengths = [
        "Demonstrated solid understanding of the topic.",
        "Maintained a helpful and professional attitude.",
        "Gave clear, practical technical advice."
    ]
    improvements = assessment["improvements"]
    seen_titles = set()
    structured_issues = []
    for imp in improvements:
        title = imp.get("title", "").strip()
        if title and title.lower() not in seen_titles:
            structured_issues.append(imp)
            seen_titles.add(title.lower())
        if len(structured_issues) >= 5:
            break

    email = f"Hi {mentor_name},\n\n"
    email += "Thank you for your recent mentorship session with Analytics Vidhya. Below is focused feedback on your adherence to mentorship guidelines.\n\n"
    email += "✅ STRENGTHS\n"
    for s in fixed_strengths:
        email += f"• {s}\n"
    email += "\n🔧 AREAS FOR IMPROVEMENT\n"
    if structured_issues:
        for idx, issue in enumerate(structured_issues, 1):
            email += f"{idx}. {issue.get('title','')}\n"
            suggestion = issue.get('suggestion','')
            if suggestion:
                email += f"   - Suggestion: {suggestion}\n\n"
    else:
        email += "No significant compliance issues observed.\n"
    email += "\nPlease review these points to align future sessions with Analytics Vidhya's standards.\n\n"
    email += "Best regards,\nAnalytics Vidhya Mentorship Review Team"
    return email


def generate_overall_summary(transcript_text: str) -> str:
    """Generate an overall session summary using LLM."""
    try:
        # Truncate transcript if too long (keep first 8000 chars for context)
        truncated_transcript = transcript_text[:8000] if len(transcript_text) > 8000 else transcript_text
        
        prompt = SESSION_SUMMARY_PROMPT.format(transcript=truncated_transcript)
        response = llm.invoke(prompt)
        summary = response.content if hasattr(response, 'content') else str(response)
        
        return summary.strip()
    except Exception as e:
        print(f"Error generating overall summary: {e}")
        return "Unable to generate overall summary due to an error."


def generate_session_checklist(transcript_text: str) -> dict:
    """Generate session checklist answers using LLM."""
    try:
        # Truncate transcript if too long
        truncated_transcript = transcript_text[:10000] if len(transcript_text) > 10000 else transcript_text
        
        prompt = SESSION_CHECKLIST_PROMPT.format(transcript=truncated_transcript)
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract JSON from response
        json_text = extract_json_from_text(response_text)
        checklist_data = json.loads(json_text)
        
        return checklist_data
    except Exception as e:
        print(f"Error generating session checklist: {e}")
        # Return default structure if error occurs
        return {
            "checklist": [
                {
                    "question": q,
                    "answer": "UNCLEAR",
                    "explanation": "Unable to analyze due to an error."
                }
                for q in [
                    "Did mentor have camera feed with Virtual Background or Blur background?",
                    "Was there any network issues or background noise?",
                    "Did mentor login on Time?",
                    "Did mentor look like he/she knows the student's profile or have asked the same from student?",
                    "Did mentor ask students what Challenges they are facing currently?",
                    "Identify the specific issue/concern by discussing with student (If Technical Session)",
                    "Commitment taken from student (On their learning time, weekly review of their own progress)",
                    "Did the mentor summarize the session, set expectations, and take clear commitments from the student on specific milestones?",
                    "Did mentor said anything negative about AV or its courses / Has mentor shared some personal details?"
                ]
            ]
        }


def format_checklist_output(checklist_data: dict) -> tuple:
    """Format checklist data into a readable string with scoring.
    
    Returns:
        tuple: (formatted_output, score_percentage)
        
    Scoring:
    - YES = 10 points
    - NO/UNCLEAR = 0 points
    - N/A (for technical question) = 10 points
    """
    output = "\n==== SESSION CHECKLIST ====\n\n"
    total_score = 0
    total_possible = 0
    
    for item in checklist_data.get("checklist", []):
        question = item.get("question", "")
        answer = item.get("answer", "UNCLEAR")
        explanation = item.get("explanation", "")
        
        # Default score calculation
        answer_upper = answer.upper()
        
        # Special case: For negative about AV question, NO is good (10 points)
        if "negative about AV" in question.lower() or "personal details" in question.lower():
            item_score = 10 if answer_upper == "NO" else 0
        # Special case: Login with UNCLEAR is 0 points
        elif "login" in question.lower() and answer_upper == "UNCLEAR":
            item_score = 0
        # Special case: N/A for technical questions is 10 points
        elif "technical" in question.lower() and answer_upper == "N/A":
            item_score = 10
        # Default case: YES is 10 points, anything else is 0
        else:
            item_score = 10 if answer_upper == "YES" else 0
        
        total_score += item_score
        total_possible += 10  # Each question is worth 10 points
        
        output += f"❓ {question}\n"
        output += f"📋 Answer: {answer}\n"
        output += f"💡 {explanation}\n"
        output += f"⭐ Score: {item_score}/10\n\n"
    
    # Calculate the score percentage
    score_percentage = (total_score / total_possible * 100) if total_possible > 0 else 0
    output += f"\n📊 CHECKLIST SCORE: {int(score_percentage)}% ({total_score}/{total_possible} points)\n"
    
    return output + "\n" + "="*50 + "\n", score_percentage


def process_transcript_enhanced(transcript_path: str, guidelines_path: str = "Guidelines.pdf") -> dict:
    with open(transcript_path, "r") as f:
        transcript_data = json.load(f)
    if isinstance(transcript_data, list):
        transcript_text = "\n".join(f"{item.get('speaker_name', 'Unknown')}: {item.get('sentence', '')}" for item in transcript_data)
    else:
        transcript_text = transcript_data.get('text', str(transcript_data))
    
    guidelines = extract_pdf_text(guidelines_path)
    chunks = chunk_text_intelligently(transcript_text)
    conversation_metrics = analyze_conversation_metrics(transcript_text)
    chunk_analyses = deep_analyze_chunks(chunks, guidelines, conversation_metrics)
    assessment = aggregate_guideline_assessment(chunk_analyses)
    mentor_name = extract_mentor_name(transcript_data, transcript_text)
    email_content = guideline_feedback_email(mentor_name, assessment)
    
    # Generate overall session summary using LLM
    print("\nGenerating overall session summary...")
    overall_summary = generate_overall_summary(transcript_text)
    
    # Generate session checklist using LLM
    print("Generating session checklist...")
    session_checklist = generate_session_checklist(transcript_text)
    
    # Format checklist for display and get the score
    checklist_output, checklist_score = format_checklist_output(session_checklist)
    
    # Ensure scores are in 0-100 range
    assessment_score = assessment.get("overall_score", 0)
    if isinstance(assessment_score, str) and assessment_score.replace('.', '').isdigit():
        assessment_score = float(assessment_score)
    elif not isinstance(assessment_score, (int, float)):
        assessment_score = 0
    assessment_score = max(0, min(100, assessment_score))
    checklist_score = max(0, min(100, checklist_score))
    
    # Calculate base score (70% assessment, 30% checklist)
    base_score = (assessment_score * 0.7) + (checklist_score * 0.3)
    
    # Apply checklist-based score ranges
    if checklist_score <= 40:
        # Scale base score to 70-80 range
        combined_score = 70 + (base_score / 100 * 10)
    elif 40 < checklist_score <= 60:
        # Scale base score to 80-85 range
        combined_score = 80 + (base_score / 100 * 5)
    else:
        # Scale base score to 85-95 range
        combined_score = 85 + (base_score / 100 * 10)
    
    # Ensure final score is within the target range
    if checklist_score <= 40:
        combined_score = max(70, min(80, combined_score))
    elif 40 < checklist_score <= 60:
        combined_score = max(80, min(85, combined_score))
    else:
        combined_score = max(85, min(95, combined_score))
    
    output = {
        "success": True,
        "overall_session_summary": overall_summary,
        "session_checklist": session_checklist,
        "overall_guideline_assessment": assessment,
        "feedback_email": email_content,
        "timestamp": datetime.datetime.now().isoformat(),
        "scores": {
            "assessment_score": round(assessment_score, 1),
            "checklist_score": round(checklist_score, 1),
            "overall_score": round(combined_score, 1)
        },
        "overall_score": round(combined_score, 1),
        "assessment_summary": {
            "professionalism": assessment.get("professionalism", "N/A"),
            "session_flow": assessment.get("session_flow", "N/A"),
            "guideline_compliance": assessment.get("overall_guideline_compliance", "N/A")
        }
    }
    
    output_path = transcript_path.replace('.json','_guideline_review.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Review Output saved at:", output_path)
    print("\n==== OVERALL SESSION SUMMARY ====\n")
    print(overall_summary)
    print("\n" + "="*50)
    print(checklist_output)
    print("="*50)
    print("\n==== EMAIL ====\n")
    print(email_content)
    print("\n" + "="*50 + "\n")
    
    return output


if __name__ == "__main__":
    import argparse
    import time
    import sys
    
    parser = argparse.ArgumentParser(description='Process mentor review transcripts.')
    parser.add_argument('--transcript', type=str, default="Gunjan-Hense-Generative-AI-Pinnacle-Program-Mentorship-97129e00-0ce6.json",
                        help='Path to the transcript JSON file')
    parser.add_argument('--guidelines', type=str, default="Guidelines.pdf",
                        help='Path to the guidelines PDF file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: print to console)')
    
    args = parser.parse_args()
    
    try:
        start_time = time.time()
        result = process_transcript_enhanced(args.transcript, args.guidelines)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
            
        print(f"\nProcessing completed in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
