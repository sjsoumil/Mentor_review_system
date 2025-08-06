import os
import streamlit as st
from dotenv import load_dotenv
from langgraph_review import process_transcript_enhanced

# Load environment variables from .env file (for local development)
load_dotenv()

# Get API key from Streamlit secrets or environment variable
openai_key = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error("OpenAI API key not found. Please set it in Streamlit secrets or .env file.")
    st.stop()

# Set the API key for OpenAI
os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit page configuration
st.set_page_config(
    page_title="Mentor Review System",
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
        st.error(f"Error saving file: {e}")
        return None

def extract_email_content(result: dict) -> str:
    """
    Given the output of process_transcript_enhanced, return the generated
    email string (or empty if not found).
    """
    # Prefer the nested "email_content" key
    if isinstance(result.get("email_content"), dict):
        return result["email_content"].get("email", "")
    # Fallback to top-level "email"
    email_field = result.get("email")
    if isinstance(email_field, dict):
        return email_field.get("email", "")
    if isinstance(email_field, str):
        return email_field
    return ""

def main():
    # "Create New" button resets the session for a fresh run
    if st.button("🆕 Create New", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.title("🎓 Mentor Review System")
    st.markdown("Upload a mentorship session transcript in JSON format to generate a detailed review.")

    # Ensure guidelines PDF exists
    GUIDELINES_PATH = "Guidelines.pdf"
    if not os.path.exists(GUIDELINES_PATH):
        with open(GUIDELINES_PATH, "w") as f:
            f.write("Standard mentorship guidelines for review process.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a transcript JSON file",
        type=["json"],
        help="Upload the transcript in JSON format"
    )

    # Step 1: After uploading, ask for date/time before processing
    if uploaded_file and not st.session_state.get("ready_to_process"):
        now = datetime.datetime.now()
        date_str = st.text_input("Enter Date (YYYY-MM-DD)", now.strftime("%Y-%m-%d"))
        time_str = st.text_input("Enter Time (HH:MM:SS)", now.strftime("%H:%M:%S"))
        if st.button("Continue", use_container_width=True):
            tmp_path = save_uploaded_file(uploaded_file)
            if not tmp_path:
                return
            st.session_state["transcript_path"] = tmp_path
            st.session_state["date_str"] = date_str
            st.session_state["time_str"] = time_str
            st.session_state["file_name"] = uploaded_file.name
            st.session_state["ready_to_process"] = True
            st.rerun()

    # Step 2: Process and display
    if st.session_state.get("ready_to_process"):
        transcript_path = st.session_state["transcript_path"]
        date_str = st.session_state["date_str"]
        time_str = st.session_state["time_str"]
        file_name = st.session_state["file_name"]

        with st.spinner("Analyzing transcript and generating review..."):
            try:
                result = process_transcript_enhanced(
                    transcript_path=transcript_path,
                    guidelines_path=GUIDELINES_PATH
                )
            except Exception as e:
                st.error(f"Processing failed: {e}")
                return
            finally:
                # cleanup temporary file
                try:
                    os.remove(transcript_path)
                except OSError:
                    pass
                st.session_state["transcript_path"] = None

        # Extract the generated email content
        email_content = extract_email_content(result)

        # Extract overall score if present
        overall_score = "N/A"
        assessment = (result.get("assessment") or result.get("final_assessment") or {})
        if isinstance(assessment, dict):
            overall_score = str(assessment.get("overall_score", "N/A"))

        # Extract mentor name for logging
        mentor_name = "Mentor"
        if email_content:
            m = re.search(r"Hi ([^,]+),", email_content)
            if m:
                mentor_name = m.group(1).strip()

        # Display feedback
        st.markdown("---")
        st.markdown(f"### 📝 Mentor Feedback")
        st.markdown(f"**Overall Score:** {overall_score}")
        st.markdown("---")

        # Read-only text area for email
        st.text_area(
            label="Feedback E-mail",
            value=email_content,
            height=350,
            disabled=True
        )

        st.download_button(
            label="📥 Download Feedback",
            data=email_content,
            file_name="mentor_feedback.txt",
            mime="text/plain",
            use_container_width=True
        )

        # Optionally append to Google Sheet
        if date_str and time_str and file_name:
            try:
                from gsheet_utils import append_feedback_row
                append_feedback_row(
                    date_str, time_str, file_name,
                    mentor_name, overall_score, email_content,
                    creds_path="GS_creds.json",
                    spreadsheet_id="1hIH41rTkaiuIWND0mbXgCtMBQBfL27KRuGaCif5dxYg"
                )
                st.success("Saved to Google Sheet!")
            except Exception as e:
                st.warning(f"Could not save to Google Sheet: {e}")
        else:
            st.warning("Missing date, time, or filename. Not saved to Google Sheet.")

        # Reset to allow next run
        st.session_state["ready_to_process"] = False

if __name__ == "__main__":
    main()
