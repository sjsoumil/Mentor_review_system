import gspread
import json
import os
from google.oauth2.service_account import Credentials
from typing import List, Dict, Any, Optional

# If you want to use a specific sheet, set this here or pass as arg
SPREADSHEET_ID = "1hIH41rTkaiuIWND0mbXgCtMBQBfL27KRuGaCif5dxYg"
SHEET_HEADER = [
    "Date", 
    "Time", 
    "Filename", 
    "Mentor Name", 
    "Overall Score", 
    "Email Output",
    "Overall Summary",
    "Checklist Summary"
]

def get_gsheet_client(creds_path: Optional[str] = None):
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive',
    ]
    
    # First try to get credentials from environment variable
    if 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
        try:
            service_account_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
            creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
            return gspread.authorize(creds)
        except Exception as e:
            print(f"Error using environment credentials: {e}")
    
    # Fall back to credentials file if specified
    if creds_path and os.path.exists(creds_path):
        try:
            creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
            return gspread.authorize(creds)
        except Exception as e:
            print(f"Error using credentials file: {e}")
    
    # Try to use default application credentials (for local development)
    try:
        from google.auth import default
        creds, _ = default(scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        print(f"Error using default credentials: {e}")
    
    raise Exception("Could not initialize Google Sheets client. No valid credentials found.")


def get_or_create_sheet_by_id(gc, spreadsheet_id: str = SPREADSHEET_ID):
    sh = gc.open_by_key(spreadsheet_id)
    worksheet = sh.sheet1
    # Ensure header
    if worksheet.row_count == 0 or worksheet.row_values(1) != SHEET_HEADER:
        worksheet.clear()
        worksheet.append_row(SHEET_HEADER)
    return worksheet


def append_feedback_row(
    date_str: str,
    time_str: str,
    filename: str,
    mentor_name: str,
    overall_score: str,
    email_output: str,
    overall_summary: str = "",
    checklist_summary: str = "",
    creds_path: Optional[str] = None,
    spreadsheet_id: str = SPREADSHEET_ID
):
    try:
        gc = get_gsheet_client(creds_path)
        worksheet = get_or_create_sheet_by_id(gc, spreadsheet_id)
        
        # Prepare the row data with all fields
        row_data = [
            date_str,
            time_str,
            filename,
            mentor_name,
            overall_score,
            email_output,
            overall_summary,
            checklist_summary
        ]
        
        worksheet.append_row(row_data)
        return True, "Successfully saved to Google Sheet"
    except Exception as e:
        error_msg = f"Could not save to Google Sheet: {str(e)}"
        print(error_msg)
        return False, error_msg
