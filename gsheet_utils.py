import gspread
from google.oauth2.service_account import Credentials
from typing import List

# If you want to use a specific sheet, set this here or pass as arg
SPREADSHEET_ID = "1hIH41rTkaiuIWND0mbXgCtMBQBfL27KRuGaCif5dxYg"
SHEET_HEADER = ["Date", "Time", "Filename", "Mentor Name", "Overall Score", "Email Output"]


def get_gsheet_client(creds_path: str = "GS_creds.json"):
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive',
    ]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc


def get_or_create_sheet_by_id(gc, spreadsheet_id: str = SPREADSHEET_ID):
    sh = gc.open_by_key(spreadsheet_id)
    worksheet = sh.sheet1
    # Ensure header
    if worksheet.row_count == 0 or worksheet.row_values(1) != SHEET_HEADER:
        worksheet.clear()
        worksheet.append_row(SHEET_HEADER)
    return worksheet


def append_feedback_row(
    date_str: str, time_str: str, filename: str, mentor_name: str, overall_score: str, email_output: str,
    creds_path: str = "GS_creds.json", spreadsheet_id: str = SPREADSHEET_ID
):
    gc = get_gsheet_client(creds_path)
    worksheet = get_or_create_sheet_by_id(gc, spreadsheet_id)
    worksheet.append_row([date_str, time_str, filename, mentor_name, overall_score, email_output])
