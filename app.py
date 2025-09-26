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
    page_title="Mentor Review System (v2)",
    page_icon="📝",
    layout="centered"
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

def extract_email_content(result: dict) -> str:
    """
    Given the output of review_system_3.process_transcript_enhanced, return the generated
    email string (or empty if not found).
    """
    # review_system_3.py returns 'feedback_email'
    email_field = result.get("feedback_email")
    if isinstance(email_field, str):
        return email_field
    # Fallbacks (in case the interface changes)
    if isinstance(result.get("email_content"), dict):
        return result["email_content"].get("email", "")
    if isinstance(result.get("email"), dict):
        return result["email"].get("email", "")
    if isinstance(result.get("email"), str):
        return result["email"]
    return ""

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

def process_single_file(transcript_path: str, guidelines_path: str, file_name: str) -> Tuple[bool, str, dict]:
    """Process a single transcript file and return results."""
    try:
        result = process_transcript_enhanced(
            transcript_path=transcript_path,
            guidelines_path=guidelines_path
        )
        email_content = extract_email_content(result)
        overall_score = extract_overall_score(result)
        return True, file_name, {"email_content": email_content, "overall_score": overall_score, "result": result}
    except Exception as e:
        return False, file_name, {"error": str(e)}

def main():
    # "Create New" button resets the session for a fresh run
    if st.button("🆕 Create New", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.title("🎓 Mentor Review System — Review Engine v2")
    st.markdown("Upload one or more mentorship session transcripts in JSON format to generate detailed reviews (using `review_system_3.py`).")

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

    # Step 1: After uploading, ask for date/time before processing
    if uploaded_files and not st.session_state.get("ready_to_process"):
        now = datetime.datetime.now()
        date_str = st.text_input("Enter Date (YYYY-MM-DD)", now.strftime("%Y-%m-%d"))
        time_str = st.text_input("Enter Time (HH:MM:SS)", now.strftime("%H:%M:%S"))
        if st.button("Process Files", use_container_width=True):
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
            st.session_state["date_str"] = date_str
            st.session_state["time_str"] = time_str
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
                        status_text.text(f"Processed: {file_name}")
                    else:
                        status_text.text(f"Error processing {file_name}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    status_text.text(f"Error processing {name}: {str(e)}")
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
        
        # Display results
        st.success(f"Processed {len([r for r in results.values() if 'error' not in r])}/{len(results)} files successfully!")
        
        # Show each result in an expander
        for file_name, result in results.items():
            with st.expander(f"Results for: {file_name}", expanded=True):
                if "error" in result:
                    st.error(f"Error processing {file_name}: {result['error']}")
                    continue
                    
                email_content = result.get("email_content", "")
                overall_score = result.get("overall_score", "N/A")
                full_result = result.get("result", {})
                
                # Extract mentor name for logging
                mentor_name = "Mentor"
                if email_content:
                    m = re.search(r"Hi ([^,]+),", email_content)
                    if m:
                        mentor_name = m.group(1).strip()
                
                st.markdown(f"### 📝 {file_name}")
                st.markdown(f"**Overall Score:** {overall_score}")
                
                # Read-only text area for email
                st.text_area(
                    label=f"Feedback for {file_name}",
                    value=email_content,
                    height=200,
                    key=f"email_{file_name}",
                    disabled=True
                )
                
                # Download button for individual feedback
                st.download_button(
                    label=f"📥 Download {file_name} Feedback",
                    data=email_content,
                    file_name=f"{os.path.splitext(file_name)[0]}_feedback.txt",
                    mime="text/plain",
                    key=f"dl_{file_name}",
                    use_container_width=True
                )
                
                # Save to Google Sheets
                try:
                    from gsheet_utils import append_feedback_row
                    success, message = append_feedback_row(
                        date_str=date_str,
                        time_str=time_str,
                        filename=file_name,
                        mentor_name=mentor_name,
                        overall_score=str(overall_score),
                        email_output=email_content,
                    )
                    if success:
                        st.success(f"Saved to Google Sheets: {message}")
                    else:
                        st.warning(f"Google Sheets: {message}")
                except Exception as e:
                    st.warning(f"Error saving to Google Sheet: {str(e)}")
                
                st.markdown("---")
        
        # Reset to allow next run
        st.session_state["ready_to_process"] = False
        st.session_state["file_paths"] = []

if __name__ == "__main__":
    main()
