import os
import json
import re
import datetime
import PyPDF2
from typing import Dict, List, TypedDict, Optional, Any
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
import statistics
import streamlit as st

# --- Ensure API Key is present ---
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    # Try to get from Streamlit secrets if not in environment
    try:
        if 'openai' in st.secrets and 'api_key' in st.secrets.openai:
            openai_key = st.secrets.openai.api_key
            os.environ["OPENAI_API_KEY"] = openai_key
    except:
        pass

if not openai_key:
    raise EnvironmentError("OpenAI API key not found. Please set it in Streamlit secrets or as an environment variable.")

# Configure Google Cloud credentials if available
if 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
    import tempfile
    from google.oauth2 import service_account
    
    try:
        # Parse the service account info from environment variable
        service_account_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
        
        # Create a temporary file with the service account info
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp:
            json.dump(service_account_info, temp)
            temp_path = temp.name
        
        # Set the environment variable that gspread will use
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_path
        
    except Exception as e:
        print(f"Warning: Failed to set up Google Cloud credentials: {e}")

def review_filename(transcript_path):
    """Generate output filename for review"""
    if transcript_path.endswith("_transcript.json"):
        base_name = transcript_path.replace("_transcript.json", "")
    elif transcript_path.endswith(".json"):
        base_name = transcript_path.replace(".json", "")
    else:
        base_name = transcript_path
    return f"{base_name}_enhanced_review.json"

def extract_pdf_text(pdf_path):
    """Extract text from PDF guidelines"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        return "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    except:
        return "Standard mentorship guidelines apply."

def extract_content(obj):
    """Extract content from LLM response"""
    if hasattr(obj, "content"):
        return obj.content
    return str(obj)

def analyze_conversation_metrics(transcript_text):
    """Analyze conversation patterns and engagement metrics"""
    lines = transcript_text.split('\n')
    mentor_lines = [line for line in lines if line.strip().lower().startswith('mentor:')]
    student_lines = [line for line in lines if line.strip().lower().startswith('student:')]
    
    # Calculate speaking ratios
    mentor_words = sum(len(line.split()) for line in mentor_lines)
    student_words = sum(len(line.split()) for line in student_lines)
    total_words = mentor_words + student_words
    
    if total_words == 0:
        return {"error": "No valid conversation found"}
    
    # Calculate engagement metrics
    mentor_questions = sum(line.count('?') for line in mentor_lines)
    student_questions = sum(line.count('?') for line in student_lines)
    
    # Calculate response lengths
    mentor_avg_response = mentor_words / max(len(mentor_lines), 1)
    student_avg_response = student_words / max(len(student_lines), 1)
    
    # Check for technical terms
    technical_terms = ['algorithm', 'data', 'model', 'python', 'code', 'api', 'database', 
                     'framework', 'library', 'function', 'variable', 'ml', 'ai', 'deep learning']
    tech_density = sum(transcript_text.lower().count(term) for term in technical_terms)
    
    return {
        "mentor_talk_ratio": mentor_words / total_words if total_words > 0 else 0.5,
        "student_talk_ratio": student_words / total_words if total_words > 0 else 0.5,
        "mentor_questions": mentor_questions,
        "student_questions": student_questions,
        "mentor_avg_response_length": mentor_avg_response,
        "student_avg_response_length": student_avg_response,
        "total_exchanges": len(mentor_lines) + len(student_lines),
        "technical_density": tech_density,
        "conversation_length": len(transcript_text.split())
    }

def chunk_text_intelligently(text, max_chars=2500):
    """Split text into meaningful chunks preserving conversation flow"""
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for line in lines:
        line_length = len(line)
        
        # Start new chunk if we exceed max_chars or find natural break
        if (current_length + line_length > max_chars and current_chunk) or \
           (line.strip().lower().startswith('mentor:') and current_length > max_chars * 0.7):
            
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        current_chunk.append(line)
        current_length += line_length
    
    # Add remaining chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def extract_json_from_text(text: str) -> str:
    """Extract JSON string from text by finding the first { and last }"""
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text

# --- Define State Schema ---
class ReviewState(TypedDict, total=False):
    transcript_text: str
    guidelines: str
    segments: list
    conversation_metrics: dict
    chunks: list
    chunk_analyses: list
    detailed_assessment: dict
    final_assessment: dict
    email_content: dict
    error: Optional[str]

# --- Enhanced Analysis Nodes ---
def initialize_enhanced_node(state: ReviewState) -> ReviewState:
    """Initialize with enhanced conversation analysis"""
    print("🔍 Initializing enhanced transcript analysis...")
    
    transcript_text = state["transcript_text"]
    
    # Get conversation metrics
    metrics = analyze_conversation_metrics(transcript_text)
    
    # Create intelligent chunks
    chunks = chunk_text_intelligently(transcript_text, max_chars=2000)
    
    print(f"📊 Conversation metrics calculated")
    print(f"📝 Split into {len(chunks)} intelligent chunks")
    
    return {
        **state,
        "conversation_metrics": metrics,
        "chunks": chunks,
        "chunk_analyses": []
    }

def deep_analyze_chunk_node(state: ReviewState) -> ReviewState:
    """Perform deep analysis of each conversation chunk"""
    chunks = state["chunks"]
    guidelines = state["guidelines"]
    metrics = state["conversation_metrics"]
    
    print(f"🔬 Starting deep analysis of {len(chunks)} chunks...")
    
    chunk_analyses = []
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    for i, chunk in enumerate(chunks):
        print(f"   Analyzing chunk {i+1}/{len(chunks)}...")
        
        analysis_prompt = PromptTemplate(
            input_variables=["chunk", "guidelines", "metrics", "chunk_number"],
            template="""
You are an expert education analyst examining this mentorship conversation chunk. Your job is to provide an honest, detailed assessment based on what actually happened in the conversation.

CONVERSATION METRICS:
{metrics}

CHUNK {chunk_number} CONTENT:
{chunk}

GUIDELINES (for reference only):
{guidelines}

Analyze this chunk thoroughly and provide a detailed assessment. Focus on:

1. ACTUAL CONVERSATION QUALITY: What really happened here?
2. MENTOR'S PERFORMANCE: How well did they handle this part?
3. STUDENT'S EXPERIENCE: What was the student's journey in this chunk?
4. LEARNING OUTCOMES: What was actually achieved?
5. SPECIFIC ISSUES: What went wrong (if anything)?

Be critical and honest. Look for:
- Vague or unhelpful responses
- Missed opportunities to help the student
- Unclear explanations
- Lack of follow-up questions
- Poor understanding of student needs
- Irrelevant advice
- Rushed explanations
- Lack of practical guidance

Also identify what worked well:
- Clear explanations
- Relevant examples
- Good questions
- Practical advice
- Student engagement
- Building confidence

Return your analysis in this JSON format:
{{
    "chunk_summary": "Brief summary of what happened in this chunk",
    "mentor_performance": {{
        "clarity_score": <1-10>,
        "relevance_score": <1-10>,
        "engagement_score": <1-10>,
        "practical_value_score": <1-10>,
        "student_understanding_score": <1-10>
    }},
    "student_experience": {{
        "confusion_level": <1-10 where 10 is very confused>,
        "engagement_level": <1-10>,
        "satisfaction_level": <1-10>,
        "learning_progress": <1-10>
    }},
    "specific_issues": [
        {{
            "issue": "Specific problem identified",
            "severity": <1-10>,
            "impact_on_learning": "How this affected the student",
            "evidence": "Specific quote or example from the conversation"
        }}
    ],
    "positive_aspects": [
        {{
            "aspect": "What worked well",
            "impact": "How this helped the student",
            "evidence": "Specific quote or example"
        }}
    ],
    "missed_opportunities": [
        "Opportunities the mentor missed to help the student better"
    ],
    "overall_chunk_quality": <1-10>,
    "learning_outcome": "What the student actually learned from this chunk",
    "mentor_adaptation": "How well the mentor adapted to the student's needs",
    "conversation_flow": "Assessment of how natural and productive the conversation was"
}}

Be honest and specific. Don't give high scores just to be nice. Base your assessment on what actually happened in the conversation.
"""
        )
        
        try:
            prompt = analysis_prompt.format(
                chunk=chunk,
                guidelines=guidelines,
                metrics=json.dumps(metrics, indent=2),
                chunk_number=i+1
            )
            
            response = llm.invoke(prompt)
            analysis_text = extract_content(response)
            
            # Extract and parse JSON
            json_text = extract_json_from_text(analysis_text)
            analysis_data = json.loads(json_text)
            
            chunk_analyses.append({
                "chunk_index": i,
                "analysis": analysis_data
            })
            
        except Exception as e:
            print(f"❌ Error analyzing chunk {i}: {str(e)}")
            # Add fallback analysis
            chunk_analyses.append({
                "chunk_index": i,
                "analysis": {
                    "chunk_summary": f"Analysis failed for chunk {i}",
                    "mentor_performance": {
                        "clarity_score": 5,
                        "relevance_score": 5,
                        "engagement_score": 5,
                        "practical_value_score": 5,
                        "student_understanding_score": 5
                    },
                    "student_experience": {
                        "confusion_level": 5,
                        "engagement_level": 5,
                        "satisfaction_level": 5,
                        "learning_progress": 5
                    },
                    "specific_issues": [{"issue": "Analysis error", "severity": 5, "impact_on_learning": "Unable to assess", "evidence": "Technical error"}],
                    "positive_aspects": [],
                    "missed_opportunities": [],
                    "overall_chunk_quality": 5,
                    "learning_outcome": "Unable to assess due to technical error",
                    "mentor_adaptation": "Unable to assess",
                    "conversation_flow": "Unable to assess"
                }
            })
    
    return {
        **state,
        "chunk_analyses": chunk_analyses
    }

def comprehensive_assessment_node(state: ReviewState) -> ReviewState:
    """Create comprehensive assessment based on detailed chunk analysis"""
    print("📋 Creating comprehensive assessment...")
    
    chunk_analyses = state["chunk_analyses"]
    metrics = state["conversation_metrics"]
    
    if not chunk_analyses:
        return {
            **state,
            "final_assessment": {"error": "No chunk analyses available"}
        }
    
    # Extract all scores and data
    clarity_scores = []
    relevance_scores = []
    engagement_scores = []
    practical_scores = []
    understanding_scores = []
    overall_quality_scores = []
    
    confusion_levels = []
    student_engagement = []
    satisfaction_levels = []
    learning_progress = []
    
    all_issues = []
    all_positives = []
    all_missed_opportunities = []
    
    # Track mentor performance metrics
    camera_backgrounds = []
    network_issues = []
    on_time_status = []
    knew_profile = []
    asked_challenges = []
    technical_issues = []
    commitments = []
    session_summaries = []
    negative_av_remarks = []
    personal_details = []
    
    # Initialize mentor criteria evaluation
    mentor_criteria = {
        "negative_av_comments": False,
        "personal_details_shared": False,
        "has_virtual_background": False,
        "mentor_on_time": True,
        "profile_reviewed": False,
        "challenges_discussed": False,
        "commitment_taken": False,
        "session_summarized": False
    }
    
    for ca in chunk_analyses:
        analysis = ca["analysis"]
        
        # Mentor performance scores
        perf = analysis.get("mentor_performance", {})
        clarity_scores.append(perf.get("clarity_score", 5))
        relevance_scores.append(perf.get("relevance_score", 5))
        engagement_scores.append(perf.get("engagement_score", 5))
        practical_scores.append(perf.get("practical_value_score", 5))
        understanding_scores.append(perf.get("student_understanding_score", 5))
        overall_quality_scores.append(analysis.get("overall_chunk_quality", 5))
        
        # Track mentor performance metrics
        camera_background = perf.get("camera_background", "not_mentioned")
        network_issue = perf.get("network_issues", "not_mentioned")
        on_time = perf.get("on_time", "not_mentioned")
        knew_student = perf.get("knew_student_profile", "not_mentioned")
        asked_challenge = perf.get("asked_about_challenges", "not_mentioned")
        technical_issue = perf.get("technical_issue_discussed", "not_mentioned")
        commitment = perf.get("commitment_taken", "not_mentioned")
        session_summary = perf.get("session_summary_provided", "not_mentioned")
        negative_av = perf.get("negative_remarks_about_av", "not_mentioned")
        personal_detail = perf.get("personal_details_shared", "not_mentioned")
        
        camera_backgrounds.append(camera_background)
        network_issues.append(network_issue)
        on_time_status.append(on_time)
        knew_profile.append(knew_student)
        asked_challenges.append(asked_challenge)
        technical_issues.append(technical_issue)
        commitments.append(commitment)
        session_summaries.append(session_summary)
        negative_av_remarks.append(negative_av)
        personal_details.append(personal_detail)
        
        # Update mentor criteria based on current chunk analysis
        if negative_av.lower() == "yes":
            mentor_criteria["negative_av_comments"] = True
        if personal_detail.lower() == "yes":
            mentor_criteria["personal_details_shared"] = True
        if camera_background.lower() in ["virtual", "blur"]:
            mentor_criteria["has_virtual_background"] = True
        if on_time.lower() == "no":
            mentor_criteria["mentor_on_time"] = False
        if knew_student.lower() == "yes":
            mentor_criteria["profile_reviewed"] = True
        if asked_challenge.lower() == "yes":
            mentor_criteria["challenges_discussed"] = True
        if commitment.lower() == "yes":
            mentor_criteria["commitment_taken"] = True
        if session_summary.lower() == "yes":
            mentor_criteria["session_summarized"] = True
        
        # Student experience
        exp = analysis.get("student_experience", {})
        confusion_levels.append(exp.get("confusion_level", 5))
        student_engagement.append(exp.get("engagement_level", 5))
        satisfaction_levels.append(exp.get("satisfaction_level", 5))
        learning_progress.append(exp.get("learning_progress", 5))
        
        # Issues and positives
        all_issues.extend(analysis.get("specific_issues", []))
        all_positives.extend(analysis.get("positive_aspects", []))
        all_missed_opportunities.extend(analysis.get("missed_opportunities", []))
    
    # Calculate averages
    avg_clarity = statistics.mean(clarity_scores)
    avg_relevance = statistics.mean(relevance_scores)
    avg_engagement = statistics.mean(engagement_scores)
    avg_practical = statistics.mean(practical_scores)
    avg_understanding = statistics.mean(understanding_scores)
    avg_quality = statistics.mean(overall_quality_scores)
    
    avg_confusion = statistics.mean(confusion_levels)  # Lower is better
    avg_student_engagement = statistics.mean(student_engagement)
    avg_satisfaction = statistics.mean(satisfaction_levels)
    avg_learning_progress = statistics.mean(learning_progress)
    
    # Calculate overall score based on multiple factors with adjusted weights
    mentor_score = (avg_clarity * 0.25 + 
                  avg_relevance * 0.25 + 
                  avg_engagement * 0.2 + 
                  avg_practical * 0.2 + 
                  avg_understanding * 0.1)
    
    student_experience_score = (avg_student_engagement * 0.4 + 
                              avg_satisfaction * 0.3 + 
                              avg_learning_progress * 0.3)
    
    # Adjust for confusion (inverse relationship)
    confusion_impact = (10 - avg_confusion) * 0.1  # 0-1 scale impact
    
    # Weight the scores with more emphasis on student experience
    base_score = (mentor_score * 0.5) + (student_experience_score * 0.5)
    
    # Scale base score to be higher (70-100 range for good sessions)
    base_score = 70 + (base_score * 3)  # 70-100 range for most sessions
    
    # Apply reduced penalties for issues
    severe_issues = [issue for issue in all_issues if issue.get("severity", 0) >= 8]  # Higher threshold for severe
    moderate_issues = [issue for issue in all_issues if 5 <= issue.get("severity", 0) < 8]
    minor_issues = [issue for issue in all_issues if issue.get("severity", 0) < 5]
    
    # Calculate penalty with reduced impact
    penalty = 0
    penalty += len(severe_issues) * 0.8    # Reduced from 1.5
    penalty += len(moderate_issues) * 0.3   # Reduced from 0.8
    penalty += len(minor_issues) * 0.1      # Reduced from 0.3
    
    # Apply conversation quality adjustments (smaller impact)
    if metrics.get("mentor_talk_ratio", 0.5) > 0.8:
        penalty += 0.3  # Reduced from 1.0
    if metrics.get("student_questions", 0) < 2 and len(chunk_analyses) > 2:
        penalty += 0.2  # Reduced from 0.5
    if metrics.get("mentor_questions", 0) < 1:
        penalty += 0.3  # Reduced from 1.0
    
    # Apply more generous bonus for positives
    bonus = min(len(all_positives) * 0.5, 5.0)  # Increased from 0.2 and cap to 5
    
    # Apply bonus for good conversation metrics
    if metrics.get("mentor_questions", 0) >= 3:
        bonus += 1.0
    if metrics.get("student_talk_ratio", 0) > 0.3:
        bonus += 0.5
    if metrics.get("technical_density", 0) > 5:  # Bonus for technical content
        bonus += min(2.0, metrics["technical_density"] * 0.2)
    
    # Calculate final score with reduced penalty impact
    final_score = max(50, min(100, base_score - penalty + bonus))  # Set floor at 50
    
    # Determine student experience rating
    student_rating = max(1, min(10, student_experience_score))
    
    # Create detailed assessment
    assessment = {
        "overall_score": round(final_score, 1),
        "student_experience_rating": round(student_rating, 1),
        "mentor_criteria": mentor_criteria,
        "detailed_scores": {
            "mentor_clarity": round(avg_clarity, 1),
            "content_relevance": round(avg_relevance, 1),
            "student_engagement": round(avg_engagement, 1),
            "practical_value": round(avg_practical, 1),
            "student_understanding": round(avg_understanding, 1),
            "overall_quality": round(avg_quality, 1)
        },
        "student_experience_breakdown": {
            "engagement_level": round(avg_student_engagement, 1),
            "satisfaction_level": round(avg_satisfaction, 1),
            "learning_progress": round(avg_learning_progress, 1),
            "confusion_level": round(avg_confusion, 1)
        },
        "conversation_analysis": {
            "mentor_talk_ratio": round(metrics.get("mentor_talk_ratio", 0.5), 2),
            "student_participation": round(metrics.get("student_talk_ratio", 0.5), 2),
            "mentor_questions_asked": metrics.get("mentor_questions", 0),
            "student_questions_asked": metrics.get("student_questions", 0),
            "total_exchanges": metrics.get("total_exchanges", 0),
            "technical_density": metrics.get("technical_density", 0)
        },
        "strengths": [
            {
                "aspect": pos.get("aspect", ""),
                "impact": pos.get("impact", ""),
                "evidence": pos.get("evidence", "")
            }
            for pos in all_positives[:8]
        ],
        "areas_for_improvement": [
            {
                "issue": issue.get("issue", ""),
                "severity": issue.get("severity", 0),
                "impact": issue.get("impact_on_learning", ""),
                "evidence": issue.get("evidence", "")
            }
            for issue in sorted(all_issues, key=lambda x: x.get("severity", 0), reverse=True)[:8]
        ],
        "missed_opportunities": all_missed_opportunities[:5],
        "assessment_summary": {
            "total_issues_found": len(all_issues),
            "severe_issues": len(severe_issues),
            "moderate_issues": len(moderate_issues),
            "positive_aspects": len(all_positives),
            "penalty_applied": round(penalty, 1),
            "bonus_applied": round(bonus, 1)
        },
        "recommendations": [],
        "chunk_summaries": [
            {
                "chunk": i + 1,
                "summary": ca["analysis"].get("chunk_summary", ""),
                "quality_score": ca["analysis"].get("overall_chunk_quality", 5),
                "learning_outcome": ca["analysis"].get("learning_outcome", "")
            }
            for i, ca in enumerate(chunk_analyses)
        ]
    }
    
    return {
        "final_assessment": assessment
    }


def generate_enhanced_email_node(state) -> dict:
    """Generate feedback email using the assessment data and return updated state"""
    print("📧 Generating mentorship feedback email...")
    
    # Handle both dictionary and object access
    if isinstance(state, dict):
        assessment = state.get("final_assessment", {})
        transcript_text = state.get("transcript_text", "")
        transcript_json = state.get("transcript_json", [])
        transcript_path = state.get('transcript_path', None) if 'transcript_path' in state else state.get('transcript_file', None)
    else:
        assessment = getattr(state, "final_assessment", {})
        transcript_text = getattr(state, "transcript_text", "")
        transcript_json = getattr(state, "transcript_json", [])
        transcript_path = getattr(state, 'transcript_path', None) if hasattr(state, 'transcript_path') else getattr(state, 'transcript_file', None)
    
    # Extract mentor name robustly from transcript JSON or text
    mentor_name = "Mentor"  # Default fallback
    student_name = None

    # Try to infer student name from file name (e.g., Swati-Eswar-...-Mentorship-....json)
    if not transcript_path:
        # Try to get from state if passed as argument
        transcript_path = state.get('transcript_file', None) if isinstance(state, dict) else None
    if transcript_path:
        import os
        base = os.path.basename(transcript_path)
        # Try to extract the student name from the filename
        parts = base.split('-')
        if len(parts) >= 2:
            possible_student = parts[0] + ' ' + parts[1]
            student_name = possible_student.strip()

    # Improved extraction from transcript_json
    if transcript_json and isinstance(transcript_json, list):
        # Count all speakers
        speakers = {}
        for turn in transcript_json:
            speaker_name = turn.get('speaker_name', '').strip()
            if speaker_name:
                speakers[speaker_name] = speakers.get(speaker_name, 0) + 1
        # If student_name is known, exclude that
        mentor_candidates = {k: v for k, v in speakers.items() if not student_name or student_name.lower() not in k.lower()}
        if len(mentor_candidates) == 1:
            mentor_name = list(mentor_candidates.keys())[0]
        elif len(mentor_candidates) > 1:
            # Pick the most frequent non-student speaker
            mentor_name = max(mentor_candidates.items(), key=lambda x: x[1])[0]
        elif speakers:
            # Fallback: pick the most frequent speaker
            mentor_name = max(speakers.items(), key=lambda x: x[1])[0]
        mentor_name = mentor_name.strip()
        print(f"✅ Extracted mentor name from JSON: {mentor_name}")

    # Fallback to text-based extraction if JSON method didn't work or yielded generic name
    if (not mentor_name or mentor_name.lower() in ["mentor", "unknown", "user", "participant"]) and transcript_text:
        lines = [line.strip() for line in transcript_text.split('\n') if line.strip()]
        name_candidates = {}
        for line in lines[:20]:
            if ':' in line:
                left = line.split(':', 1)[0].strip()
                # Exclude student name and generic roles
                if (not student_name or student_name.lower() not in left.lower()) and left.lower() not in ['mentor', 'student', 'participant', 'user', 'unknown'] and len(left.split()) <= 3:
                    name_candidates[left] = name_candidates.get(left, 0) + 1
        if name_candidates:
            mentor_name = max(name_candidates.items(), key=lambda x: x[1])[0]
            print(f"✅ Extracted mentor name from text: {mentor_name}")

    # Final fallback: capitalize if still generic
    if not mentor_name or mentor_name.lower() in ["mentor", "unknown", "user", "participant"]:
        mentor_name = None

    # Final fallback for student_name
    if not student_name or student_name.lower() in ["student", "unknown", "user", "participant"]:
        student_name = None

    # If still not found, fallback to generic
    mentor_name = mentor_name or "Mentor"
    student_name = student_name or "Unknown"
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    # Get and refine strengths
    strengths = assessment.get("strengths", [
        "Demonstrated strong technical knowledge",
        "Maintained professional and supportive demeanor",
        "Provided clear and actionable guidance"
    ])
    
    # Prepare strengths for refinement
    strengths_text = "\n".join([f"- {s}" for s in strengths[:3]])
    
    # Format strengths to focus on mentor's performance
    refined_strengths = [
        "Demonstrated solid understanding of the topic.",
        "Maintained a helpful and professional attitude",
        "Gave clear, practical technical advice"
    ]
    #   # Refine strengths using LLM
    # strengths_prompt = f"""Refine these mentor strengths to be more specific and impactful:
    
    # {strengths_text} 
    
    # Make them more specific to the session and focus on the mentor's actions and their positive impact.
    # Keep each point concise (1-2 sentences).
    # """
    
    # try:
    #     refined_strengths_response = llm.invoke(strengths_prompt)
    #     refined_strengths = [s.strip() for s in refined_strengths_response.content.split('\n') if s.strip()]
    # except Exception as e:
    #     print(f"❌ Error refining strengths: {str(e)}")
    #     refined_strengths = [s for s in strengths[:3]]
    # Get and refine areas for improvement
    improvements = assessment.get("areas_for_improvement", [])
    
    # Prepare improvements for refinement
    improvement_text = ""
    if improvements:
        for imp in improvements:  # Limit to top 5 improvements
            issue = imp.get("issue", "")
            impact = imp.get("impact", "")
            if issue and impact:
                improvement_text += f"- {issue}: {impact}\n"
    
    # Refine improvements using LLM
    if improvement_text:
        improvement_prompt = f"""Refine these areas for improvement to be more constructive and actionable  and all summarised in 4-5 points with each point no more than 30 words only but make sure you do not provide any generic feedback:
        
        {improvement_text}
        
        Make the feedback:
        1. Specific and focused on observable behaviors
        2. Include the impact of the behavior
        3. Suggest concrete ways to improve
        4. Maintain a supportive, growth-oriented tone
        5.Make it is concise and impactful
        6.Maintained a professional and supportive demeanor 
        
        Format each as a single, concise bullet point.
        """
        
        try:
            refined_improvements_response = llm.invoke(improvement_prompt)
            refined_improvements = [i.strip() for i in refined_improvements_response.content.split('\n') if i.strip()]
        except Exception as e:
            print(f"❌ Error refining improvements: {str(e)}")
            refined_improvements = [f"{i+1}. {imp}" for i, imp in enumerate(improvement_text.split('\n')[:5]) if imp.strip()]
    else:
        refined_improvements = [
            "1. Consider increasing engagement through more interactive questioning",
            "2. Provide more detailed technical guidance and examples",
            "3. Take time to better understand each student's specific context"
            "4. These points should be concise and impactful"
        ]
    
    # Build email with personalized greeting
    
    email = f"Hi {mentor_name},\n\n"
    email += "Thank you for your recent mentorship session. Following our review, we are sharing key feedback to enhance your mentorship effectiveness.\n\n"
    
    # Add refined strengths with proper formatting
    email += "\n✅ STRENGTHS\n\n"
    
    for i, strength in enumerate(refined_strengths[:3], 1):
        # Remove any existing bullet points
        strength = re.sub(r'^\s*[-•]\s*', '', str(strength)).strip()
        email += f"• {strength}\n"
    email += "\n"  # Add extra newline after strengths
    
    # Add refined improvements with numbered points and proper spacing
    email += "\n1️⃣ KEY AREAS FOR IMPROVEMENT\n\n"
    
    for i, improvement in enumerate(refined_improvements, 1):
        # Remove any existing numbering or bullet points
        improvement = re.sub(r'^\s*[0-9]+\.?\s*', '', improvement)  # Remove numbers
        improvement = re.sub(r'^\s*[-•]\s*', '', improvement)  # Remove bullets
        improvement = improvement.strip()
        
        # Format as numbered point with proper spacing
        email += f"{i}. {improvement}\n\n"  # Double newline for better spacing
    
    # Closing
    email += "\nPlease review these points and consider how you might address them in your upcoming sessions. We're here to support your growth as a mentor.\n\n"
    email += "Best regards,\nMentorship Review Team"
    
    # Save email to a separate file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    email_filename = f"mentor_feedback_{mentor_name.lower()}_{timestamp}.txt"
    with open(email_filename, 'w') as f:
        f.write(email)
    print(f"📨 Email saved to: {email_filename}")
    
    # Return the updated state with email content
    if isinstance(state, dict):
        state["email_content"] = {"email": email}
        state["mentor_name"] = mentor_name
        state["student_name"] = student_name
    else:
        setattr(state, "email_content", {"email": email})
        setattr(state, "mentor_name", mentor_name)
        setattr(state, "student_name", student_name)
    
    return state

# --- Build Enhanced LangGraph ---
def build_enhanced_review_graph():
    """Build the enhanced LangGraph workflow"""
    workflow = StateGraph(ReviewState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_enhanced_node)
    workflow.add_node("deep_analyze", deep_analyze_chunk_node)
    workflow.add_node("comprehensive_assess", comprehensive_assessment_node)
    workflow.add_node("generate_email", generate_enhanced_email_node)
    
    # Define the flow
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "deep_analyze")
    workflow.add_edge("deep_analyze", "comprehensive_assess")
    workflow.add_edge("comprehensive_assess", "generate_email")
    workflow.add_edge("generate_email", END)
    
    return workflow.compile()

# --- Main Processing Function ---
def process_transcript_enhanced(transcript_path: str, guidelines_path: str = "Guidelines.pdf"):
    """Process a transcript file with enhanced analysis"""
    
    # Load transcript
    try:
        with open(transcript_path, "r") as f:
            transcript_data = json.load(f)
            
        # Handle the actual transcript format
        if isinstance(transcript_data, list):
            # Convert list of segments to text
            transcript_text = "\n".join(
                f"{item.get('speaker_name', 'Unknown')}: {item.get('sentence', '')}" 
                for item in transcript_data
            )
            segments = transcript_data
        else:
            # Handle other formats if needed
            transcript_text = transcript_data.get('text', str(transcript_data))
            segments = transcript_data.get('segments', [])
            
    except Exception as e:
        print(f"❌ Error loading transcript: {str(e)}")
        return None
    
    # Load guidelines
    guidelines_text = extract_pdf_text(guidelines_path)
    
    # Initialize state
    initial_state = {
        "transcript_text": transcript_text,
        "transcript_json": segments, #transcript_data if isinstance(transcript_data, list) else [],
        "guidelines": guidelines_text,
        "segments": segments
    }
    
    # Build and run the graph
    graph = build_enhanced_review_graph()
    
    try:
        print(f"🚀 Starting enhanced analysis of: {transcript_path}")
        final_state = graph.invoke(initial_state)
        
        # Generate email content
        email_content = ""
        try:
            # Ensure final_assessment is a dictionary
            final_assessment = final_state.get("final_assessment", {})
            if not isinstance(final_assessment, dict):
                final_assessment = {}
                
            # Create a ReviewState object with all required fields
            review_state = {
                "transcript_text": final_state.get("transcript_text", ""),
                "transcript_json": final_state.get("transcript_json", []),
                "guidelines": final_state.get("guidelines", ""),
                "segments": final_state.get("segments", []),
                "chunks": [],
                "chunk_analyses": [],
                "conversation_metrics": final_state.get("conversation_metrics", {}),
                "detailed_assessment": {},
                "final_assessment": final_assessment,
                "email_content": {},
                "error": None
            }
            
            # Generate email content
            if isinstance(review_state, dict):
                # update review_state in place and then extract the actual email string
                review_state = generate_enhanced_email_node(review_state)
                email_content = review_state.get("email_content", {}).get("email", "")
            else:
                email_content = str(review_state)
                
        except Exception as e:
            print(f"❌ Error generating email: {str(e)}")
            email_content = "Error generating email content. Please check the detailed assessment for feedback."
        
        # Prepare output with email content
        output = {
            "success": True,
            "analysis_type": "enhanced_deep_analysis",
            "conversation_metrics": final_state.get("conversation_metrics", {}),
            "assessment": final_state.get("final_assessment", {}),
            "email": email_content,  # Direct email content
            "error": final_state.get("error"),
            "timestamp": datetime.datetime.now().isoformat(),
            "analysis_approach": "comprehensive_conversation_analysis"
        }
        
        # Save output
        output_path = review_filename(transcript_path)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✅ Enhanced analysis completed! Output saved to: {output_path}")
        return output
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }
# --- Main Execution ---
if __name__ == "__main__":
    # Specify the transcript file here
    transcript_path = "Gunjan-Hense-Generative-AI-Pinnacle-Program-Mentorship-97129e00-0ce6.json"
    guidelines_path = "Guidelines.pdf"
    
    # Process the transcript
    result = process_transcript_enhanced(transcript_path, guidelines_path)
    
    if result and result.get("success"):
        assessment = result.get("assessment", {})
        print(f"\n🎯 ENHANCED ANALYSIS COMPLETE")
        print(f"📊 Overall Score: {assessment.get('overall_score', 'N/A')}/100")
        print(f"👤 Student Experience: {assessment.get('student_experience_rating', 'N/A')}/10")
        print(f"📈 Issues Found: {assessment.get('assessment_summary', {}).get('total_issues_found', 'N/A')}")
        print(f"✨ Strengths Identified: {len(assessment.get('strengths', []))}")
        print(f"🔧 Areas for Improvement: {len(assessment.get('areas_for_improvement', []))}")
    else:
        print(f"\n❌ PROCESSING FAILED")
        if result:
            print(f"Error: {result.get('error', 'Unknown error')}")
