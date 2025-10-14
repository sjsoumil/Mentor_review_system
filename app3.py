import os
import tempfile
import datetime
import re
import concurrent.futures
from typing import Tuple

import streamlit as st
from review_system_3 import process_transcript_enhanced

# Load environment variables from Streamlit secrets
try:
    # Set OpenAI API key from Streamlit secrets
    if 'openai' in st.secrets and 'api_key' in st.secrets.openai:
        os.environ["OPENAI_API_KEY"] = st.secrets.openai.api_key
    # Set Google Cloud credentials from Streamlit secrets
    if 'gcp_service_account' in st.secrets:
        # Convert the service account dict to JSON string for gspread
        import json
        service_account_info = dict(st.secrets.gcp_service_account)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = json.dumps(service_account_info)
    
    # Verify OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key not found in Streamlit secrets. Please configure it in the Streamlit Cloud settings.")
        st.stop()
        
except Exception as e:
    st.error(f"Error loading configuration: {str(e)}")
    st.stop()

# Streamlit page configuration
st.set_page_config(
    page_title="Mentor Review System",
    page_icon="📝",
    layout="wide"
)

def save_uploaded_file(uploaded_file) -> str | None:
    """Save uploaded file to a temporary JSON and return its path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name
    except Exception as e:
        st.error(f"Error saving file {uploaded_file.name}: {e}")
        return None

def extract_email_content(result: dict) -> Tuple[str, str, str, dict]:
    """
    Given the output of review_system_3.process_transcript_enhanced, return a tuple of
    (email_content, overall_summary, checklist_formatted, checklist_data) strings.
    """
    # Get email content
    email_field = result.get("feedback_email", "")
    if not isinstance(email_field, str):
        # Fallbacks (in case the interface changes)
        if isinstance(result.get("email_content"), dict):
            email_field = result["email_content"].get("email", "")
        elif isinstance(result.get("email"), dict):
            email_field = result["email"].get("email", "")
        elif isinstance(result.get("email"), str):
            email_field = result["email"]
        else:
            email_field = ""
    
    # Get overall session summary (NEW)
    overall_summary = result.get("overall_session_summary", "")
    
    # Get session checklist (NEW)
    checklist_data = result.get("session_checklist", {})
    
    # Format checklist for display
    checklist_formatted = format_checklist_display(checklist_data)
    
    return email_field, overall_summary, checklist_formatted, checklist_data

def format_checklist_display(checklist_data: dict) -> str:
    """Format checklist data into a readable string."""
    if not checklist_data or "checklist" not in checklist_data:
        return "No checklist data available."
    
    output = ""
    for item in checklist_data.get("checklist", []):
        question = item.get("question", "")
        answer = item.get("answer", "UNCLEAR")
        explanation = item.get("explanation", "")
        
        # Add emoji based on answer
        if answer.upper() == "YES":
            emoji = "✅"
        elif answer.upper() == "NO":
            emoji = "❌"
        else:
            emoji = "❓"
        
        output += f"{emoji} **{question}**\n"
        output += f"**Answer:** {answer}\n"
        output += f"{explanation}\n\n"
    
    return output

def extract_overall_score(result: dict) -> str:
    """
    Extract overall score consistently for review_system_3 output.
    """
    # Direct key if present
    score = result.get("overall_score")
    if isinstance(score, (int, float)):
        return str(score)
    # Nested inside overall_guideline_assessment
    oga = result.get("overall_guideline_assessment") or {}
    score = oga.get("overall_score")
    if isinstance(score, (int, float)):
        return str(score)
    return "N/A"

def extract_detailed_scores(result: dict) -> dict:
    """Extract detailed guideline scores including checklist score."""
    scores = {}
    
    # Get main assessment scores
    oga = result.get("overall_guideline_assessment", {})
    detailed = oga.get("detailed_scores", {})
    
    # Add standard scores
    scores.update({
        "Guideline Professionalism": detailed.get("guideline_professionalism", "N/A"),
        "Guideline Session Flow": detailed.get("guideline_session_flow", "N/A"),
        "Guideline Compliance": detailed.get("guideline_compliance", "N/A"),
    })
    
    # Add checklist score if available
    if "scores" in result and "checklist_score" in result["scores"]:
        checklist_score = result["scores"]["checklist_score"]
        scores["Checklist Score"] = f"✓ {checklist_score:.1f}"
    
    return scores

def process_single_file(transcript_path: str, guidelines_path: str, file_name: str) -> Tuple[bool, str, dict]:
    """Process a single transcript file and return results."""
    try:
        result = process_transcript_enhanced(
            transcript_path=transcript_path,
            guidelines_path=guidelines_path
        )
        email_content, overall_summary, checklist_formatted, checklist_data = extract_email_content(result)
        overall_score = extract_overall_score(result)
        detailed_scores = extract_detailed_scores(result)
        
        return True, file_name, {
            "email_content": email_content,
            "overall_summary": overall_summary,
            "checklist_formatted": checklist_formatted,
            "checklist_data": checklist_data,
            "overall_score": overall_score,
            "detailed_scores": detailed_scores,
            "result": result
        }
    except Exception as e:
        return False, file_name, {"error": str(e)}

def main():
    # Add custom CSS for better UI
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 20px;
        }
        .stProgress > div > div > div > div {
            background-color: #1E88E5;
        }
        .stAlert {
            border-radius: 10px;
        }
        .score-card {
            padding: 1rem;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin: 0.5rem 0;
        }
        .metric-container {
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'ready_to_process' not in st.session_state:
        st.session_state.ready_to_process = False
    if 'file_paths' not in st.session_state:
        st.session_state.file_paths = []

    # Sidebar with app info
    with st.sidebar:
        st.title("ℹ️ About")
        st.markdown("""
        **Mentor Review System ** analyzes mentorship sessions 
        based on guideline adherence.
        
        ### Features:
        - 📊 Guideline-based scoring
        - 📝 Overall session summary
        - ✅ Session checklist
        - 📧 Automated feedback emails
        - 📈 Detailed compliance metrics
        
        ### How to use:
        1. Upload transcript JSON files
        2. Enter session date/time
        3. Click "Process Files"
        4. View and download results
        
        ---
        
        **Note:** Scoring is based **only** on guideline adherence, 
        not on technical knowledge or teaching quality.
        """)
        
        # Add reset button in the sidebar
        if st.button("🔄 Start New Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # Main content
    st.title("🎓 Mentor Review System")
    st.markdown("""
    Upload mentorship session transcripts to generate comprehensive reviews based on 
    **Analytics Vidhya mentorship guidelines**. All scores reflect guideline compliance only.
    """)

    # Ensure guidelines PDF exists
    GUIDELINES_PATH = "Guidelines.pdf"
    if not os.path.exists(GUIDELINES_PATH):
        with open(GUIDELINES_PATH, "w") as f:
            f.write("Standard mentorship guidelines for review process.")

    # File uploader - now accepts multiple files
    uploaded_files = st.file_uploader(
        "Choose transcript JSON files",
        type=["json"],
        help="Upload one or more transcript files in JSON format",
        accept_multiple_files=True
    )

    # Initialize date and time in session state if not exists
    if 'date_str' not in st.session_state:
        now = datetime.datetime.now()
        st.session_state['date_str'] = now.strftime("%Y-%m-%d")
    if 'time_str' not in st.session_state:
        now = datetime.datetime.now()
        st.session_state['time_str'] = now.strftime("%H:%M:%S")

    # Step 1: After uploading, ask for date/time before processing
    if uploaded_files and not st.session_state.get("ready_to_process"):
        col1, col2 = st.columns(2)
        with col1:
            date_str = st.text_input("Enter Date (YYYY-MM-DD)", 
                                   value=st.session_state['date_str'],
                                   key="date_input")
        with col2:
            time_str = st.text_input("Enter Time (HH:MM:SS)", 
                                   value=st.session_state['time_str'],
                                   key="time_input")
        
        # Update session state when inputs change
        st.session_state['date_str'] = date_str
        st.session_state['time_str'] = time_str
        
        if st.button("🚀 Process Files", use_container_width=True, type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
                return
                
            file_paths = []
            for file in uploaded_files:
                tmp_path = save_uploaded_file(file)
                if tmp_path:
                    file_paths.append((tmp_path, file.name))
            
            if not file_paths:
                st.error("Failed to save one or more files. Please try again.")
                return
                
            st.session_state["file_paths"] = file_paths
            st.session_state["ready_to_process"] = True
            st.rerun()

    # Step 2: Process and display
    if st.session_state.get("ready_to_process"):
        file_paths = st.session_state.get("file_paths", [])
        date_str = st.session_state["date_str"]
        time_str = st.session_state["time_str"]
        
        if not file_paths:
            st.error("No valid files to process.")
            st.session_state["ready_to_process"] = False
            return
            
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the load operations and mark each future with its file name
            future_to_file = {
                executor.submit(process_single_file, path, GUIDELINES_PATH, name): (path, name)
                for path, name in file_paths
            }
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                path, name = future_to_file[future]
                try:
                    success, file_name, result = future.result()
                    results[file_name] = result
                    if success:
                        status_text.text(f"✅ Processed: {file_name}")
                    else:
                        status_text.text(f"❌ Error processing {file_name}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    status_text.text(f"❌ Error processing {name}: {str(e)}")
                    results[name] = {"error": str(e)}
                
                # Update progress
                progress = (i + 1) / len(file_paths)
                progress_bar.progress(min(progress, 1.0))
        
        # Clean up temporary files
        for path, _ in file_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        
        # Display summary
        successful = len([r for r in results.values() if 'error' not in r])
        st.success(f"✅ Successfully processed {successful}/{len(results)} files!")
        
        # Show each result in an expander
        for file_name, result in results.items():
            with st.expander(f"📄 {file_name}", expanded=True):
                if "error" in result:
                    st.error(f"Error processing {file_name}: {result['error']}")
                    continue
                    
                email_content = result.get("email_content", "")
                overall_summary = result.get("overall_summary", "")
                checklist_formatted = result.get("checklist_formatted", "")
                checklist_data = result.get("checklist_data", {})
                overall_score = result.get("overall_score", "N/A")
                detailed_scores = result.get("detailed_scores", {})
                full_result = result.get("result", {})
                
                # Extract mentor name for logging
                mentor_name = "Mentor"
                if email_content:
                    m = re.search(r"Hi ([^,]+),", email_content)
                    if m:
                        mentor_name = m.group(1).strip()
                
                # Header with file name
                st.markdown(f"### 📝 {file_name}")
                
                # Display scores in columns
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Overall Score", f"{overall_score}/100")
                with col2:
                    prof_score = detailed_scores.get("Guideline Professionalism", "N/A")
                    st.metric("Professionalism", f"{prof_score}/10" if prof_score != "N/A" else prof_score)
                with col3:
                    flow_score = detailed_scores.get("Guideline Session Flow", "N/A")
                    st.metric("Session Flow", f"{flow_score}/10" if flow_score != "N/A" else flow_score)
                with col4:
                    comp_score = detailed_scores.get("Guideline Compliance", "N/A")
                    st.metric("Compliance", f"{comp_score}/10" if comp_score != "N/A" else comp_score)
                with col5:
                    checklist_score = detailed_scores.get("Checklist Score", "N/A")
                    st.metric("Checklist", f"{checklist_score}" if checklist_score != "N/A" else checklist_score, 
                            help="Checklist score based on session requirements (0-100)")
                
                st.markdown("---")
                
                # Tabs for different sections
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📧 Feedback Email", 
                    "📊 Overall Summary", 
                    "✅ Session Checklist",
                    "📥 Downloads"
                ])
                
                with tab1:
                    st.markdown("#### Feedback Email for Mentor")
                    st.text_area(
                        label=f"Feedback for {file_name}",
                        value=email_content,
                        height=400,
                        key=f"email_{file_name}",
                        disabled=True,
                        label_visibility="collapsed"
                    )
                
                with tab2:
                    st.markdown("#### Overall Session Summary")
                    if overall_summary:
                        st.markdown(overall_summary)
                    else:
                        st.info("No overall summary available for this transcript.")
                
                with tab3:
                    st.markdown("#### Session Checklist")
                    if checklist_formatted:
                        st.markdown(checklist_formatted)
                    else:
                        st.info("No checklist data available for this transcript.")
                
                with tab4:
                    st.markdown("#### Download Options")
                    
                    # Create download buttons
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    
                    with col_dl1:
                        st.download_button(
                            label="📥 Download Feedback",
                            data=email_content,
                            file_name=f"{os.path.splitext(file_name)[0]}_feedback.txt",
                            mime="text/plain",
                            key=f"dl_feedback_{file_name}",
                            use_container_width=True
                        )
                    
                    with col_dl2:
                        st.download_button(
                            label="📥 Download Summary",
                            data=overall_summary,
                            file_name=f"{os.path.splitext(file_name)[0]}_summary.txt",
                            mime="text/plain",
                            key=f"dl_summary_{file_name}",
                            use_container_width=True
                        )
                    
                    with col_dl3:
                        st.download_button(
                            label="📥 Download Checklist",
                            data=checklist_formatted,
                            file_name=f"{os.path.splitext(file_name)[0]}_checklist.txt",
                            mime="text/plain",
                            key=f"dl_checklist_{file_name}",
                            use_container_width=True
                        )
                    
                    # Full JSON download
                    import json
                    st.download_button(
                        label="📥 Download Complete Report (JSON)",
                        data=json.dumps(full_result, indent=2),
                        file_name=f"{os.path.splitext(file_name)[0]}_complete_report.json",
                        mime="application/json",
                        key=f"dl_json_{file_name}",
                        use_container_width=True
                    )
                
                # Save to Google Sheets
                try:
                    from gsheet_utils import append_feedback_row
                    
                    # Prepare checklist summary for Google Sheets
                    checklist_summary = ""
                    if checklist_data and "checklist" in checklist_data:
                        for item in checklist_data["checklist"]:
                            q = item.get("question", "")
                            a = item.get("answer", "")
                            checklist_summary += f"{q}: {a}\n"
                    
                    success, message = append_feedback_row(
                        date_str=date_str,
                        time_str=time_str,
                        filename=file_name,
                        mentor_name=mentor_name,
                        overall_score=str(overall_score),
                        email_output=email_content,
                        overall_summary=overall_summary,
                        checklist_summary=checklist_summary
                    )
                    if success:
                        st.success(f"✅ Saved to Google Sheets: {message}")
                    else:
                        st.warning(f"⚠️ Google Sheets: {message}")
                except Exception as e:
                    st.warning(f"⚠️ Error saving to Google Sheet: {str(e)}")
                
                st.markdown("---")
        
        # Reset to allow next run
        st.session_state["ready_to_process"] = False
        st.session_state["file_paths"] = []

if __name__ == "__main__":
    main()
