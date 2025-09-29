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
    {{"guideline": "Which point was followed", "example": "Example phrase or behavior"}}
  ],
  "guideline_violations": [
    {{
      "guideline": "Which guideline was NOT followed",
      "severity": <int 1-10>,
      "evidence": "Exact phrase/example from chunk",
      "impact": "Why this matters for mentorship quality"
    }}
  ],
  "improvement_suggestions": [
    {{
      "title": "[Specific area for improvement]",
      "suggestion": "[Actionable recommendation phrased as constructive feedback. Use second person 'you' and be specific about what to improve and how. For example: 'Consider pausing after asking questions to give students more time to respond. A 3-5 second wait time can encourage more thoughtful responses.' Keep it professional and supportive.]"
    }}
  ],
  "professionalism": <int 1-10>,
  "session_flow": <int 1-10>,
  "overall_guideline_compliance": <int 1-10>
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
                "professionalism": 5,
                "session_flow": 5,
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
    """Aggregate scores and feedback from chunk analyses using a discriminative rubric
    and apply a gentle calibration so typical sessions land near 75/100 on average.
    """
    import statistics

    # Trackers
    clarity_scores, relevance_scores, engagement_scores = [], [], []
    practical_scores, understanding_scores = [], []
    overall_quality_scores, professionalism_scores, session_flow_scores = [], [], []
    all_positive_behaviors, all_violations, all_improvements = [], [], []
    all_issues, all_missed_opportunities = [], []
    all_learning_outcomes, all_mentor_adaptations, all_conversation_flows = [], [], []

    for ca in (chunk_analyses or []):
        analysis = (ca or {}).get("analysis", {})

        perf = analysis.get("mentor_performance", {})
        if isinstance(perf, dict):
            for key, bucket in [
                ("clarity_score", clarity_scores),
                ("relevance_score", relevance_scores),
                ("engagement_score", engagement_scores),
                ("practical_value_score", practical_scores),
                ("student_understanding_score", understanding_scores),
            ]:
                v = perf.get(key)
                if isinstance(v, (int, float)):
                    bucket.append(float(v))

        student = analysis.get("student_experience", {})
        if isinstance(student, dict):
            v1 = student.get("satisfaction_level")
            v2 = student.get("engagement_level")
            if isinstance(v1, (int, float)): professionalism_scores.append(float(v1))
            if isinstance(v2, (int, float)): session_flow_scores.append(float(v2))

        oq = analysis.get("overall_chunk_quality")
        if isinstance(oq, (int, float)):
            overall_quality_scores.append(float(oq))

        all_positive_behaviors.extend(analysis.get("positive_aspects", []) or [])
        all_violations.extend(analysis.get("guideline_violations", []) or [])
        all_issues.extend(analysis.get("specific_issues", []) or [])
        for imp in analysis.get("improvement_suggestions", []) or []:
            if isinstance(imp, str):
                imp = {"title": imp, "suggestion": imp}
            if imp and imp not in all_improvements:
                all_improvements.append(imp)

        if "missed_opportunities" in analysis:
            all_missed_opportunities.extend(analysis.get("missed_opportunities") or [])
        if "learning_outcome" in analysis:
            all_learning_outcomes.append(analysis["learning_outcome"])
        if "mentor_adaptation" in analysis:
            all_mentor_adaptations.append(analysis["mentor_adaptation"])
        if "conversation_flow" in analysis:
            all_conversation_flows.append(analysis["conversation_flow"])

    n_chunks = max(1, len(chunk_analyses or []))

    def _sev(item, default=5.0):
        try:
            return float(item.get("severity", default))
        except Exception:
            return float(default)

    severe_issues = [i for i in all_issues if _sev(i) >= 8]
    moderate_issues = [i for i in all_issues if 5 <= _sev(i) < 8]

    # Normalize counts by chunks
    pos_rate = len(all_positive_behaviors) / n_chunks
    sev_rate = len(severe_issues) / n_chunks
    mod_rate = len(moderate_issues) / n_chunks
    vio_rate = len(all_violations) / n_chunks

    # Heuristic metric if missing/flat
    def _heuristic():
        # Tilted towards positive (baseline 6.8) so average centers closer to 7.5 after calibration
        base = 6.8 + 0.7 * pos_rate - 0.7 * sev_rate - 0.4 * mod_rate - 0.2 * vio_rate
        return max(1.0, min(10.0, base))

    def _robust_mean(scores):
        if not scores:
            return _heuristic()
        m = statistics.mean(scores)
        sd = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        # Replace flat near-5 with heuristic
        if abs(m - 5.0) < 0.2 and sd < 0.2:
            return _heuristic()
        return max(1.0, min(10.0, m))

    avg_scores = {
        "clarity": _robust_mean(clarity_scores),
        "relevance": _robust_mean(relevance_scores),
        "engagement": _robust_mean(engagement_scores),
        "practical_value": _robust_mean(practical_scores),
        "student_understanding": _robust_mean(understanding_scores),
        "overall_quality": _robust_mean(overall_quality_scores),
        "professionalism": _robust_mean(professionalism_scores),
        "session_flow": _robust_mean(session_flow_scores),
    }

    weights = {
        "clarity": 0.15,
        "relevance": 0.15,
        "engagement": 0.15,
        "practical_value": 0.15,
        "student_understanding": 0.10,
        "professionalism": 0.10,
        "session_flow": 0.10,
        "overall_quality": 0.10,
    }

    base_score = sum(avg_scores[k] * w for k, w in weights.items())  # 1-10

    # Penalty/bonus (gentler penalty, slightly stronger bonus)
    sev_mean = statistics.mean([_sev(i) for i in all_issues]) if all_issues else 0.0
    penalty = min(2.0, 0.6 * sev_rate + 0.3 * mod_rate + 0.05 * (sev_mean / 10.0))
    bonus = min(2.5, 0.6 * pos_rate)

    # Consistency credit
    try:
        core = [avg_scores["clarity"], avg_scores["relevance"], avg_scores["engagement"], avg_scores["practical_value"]]
        consistency = statistics.pstdev(core)
        if statistics.mean(core) >= 7.5 and consistency < 0.6:
            bonus += 0.4
    except Exception:
        pass

    raw = base_score + bonus - penalty  # 1-10 approx

    # --- Calibration to average ≈ 7.5 (75/100) ---
    # Assume a 'neutral' session raw ~= 6.8; scale changes slightly to spread
    neutral_point = 6.8
    scale = 1.1
    calibrated = 7.5 + (raw - neutral_point) * scale
    final_0_10 = max(1.0, min(10.0, calibrated))

    scaled_score = round(final_0_10 * 10.0, 1)  # 0-100

    summary = ""
    if all_learning_outcomes:
        summary = ". ".join([s for s in all_learning_outcomes if s])

    return {
        "overall_score": scaled_score,
        "detailed_scores": {k: round(v, 1) for k, v in avg_scores.items()},
        "professionalism": round(avg_scores["professionalism"], 1),
        "session_flow": round(avg_scores["session_flow"], 1),
        "overall_guideline_compliance": round(avg_scores["overall_quality"], 1),
        "positive_behaviors": all_positive_behaviors,
        "violations": all_violations,
        "improvements": all_improvements,
        "issues": all_issues,
        "missed_opportunities": all_missed_opportunities,
        "summaries": [s for s in [summary] if s],
        "mentor_adaptations": all_mentor_adaptations,
        "conversation_flows": all_conversation_flows,
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


@lru_cache(maxsize=32)
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
    output = {
        "success": True,
        "overall_guideline_assessment": assessment,
        "feedback_email": email_content,
        "timestamp": datetime.datetime.now().isoformat(),
        "overall_score": assessment.get("overall_score", "N/A"),
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
    print("\n==== EMAIL ====\n")
    print(email_content)
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
