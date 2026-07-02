import time
import requests
import re
import os
import google.generativeai as genai
from bs4 import BeautifulSoup
import json

# --- Configuration ---
# IMPORTANT: For security, your specific IP address and credentials have been replaced with placeholders.
# Please update these variables with your actual details before running the script.
MOODLE_BASE_URL = "http://103.117.208.19/moodle"
FILES_URL = f"{MOODLE_BASE_URL}/user/files.php"
LOGIN_URL = f"{MOODLE_BASE_URL}/login/index.php"

USERNAME = os.getenv("MOODLE_USERNAME", "2402111144") # Fallback to default if not set
PASSWORD = os.getenv("MOODLE_PASSWORD")
AI_API_KEY = os.getenv("AI_API_KEY")

if not PASSWORD or not AI_API_KEY:
    print("Error: MOODLE_PASSWORD and AI_API_KEY environment variables must be set.")
    exit(1)

# Configure the AI Studio API (using the Google GenAI SDK as an example)
genai.configure(api_key=AI_API_KEY)
model = genai.GenerativeModel('gemini-3.1-flash-lite')

def get_login_token(html_content):
    """Extracts the Moodle logintoken from the login page HTML."""
    match = re.search(r'name="logintoken" value="([^"]+)"', html_content)
    if match:
        return match.group(1)
    return None

def extract_filemanager_params(html_content):
    """Extracts required tokens from the Moodle files page."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    sesskey_input = soup.find('input', {'name': 'sesskey'})
    sesskey = sesskey_input.get('value') if sesskey_input else None
    if not sesskey:
        sesskey_match = re.search(r'"sesskey"\s*:\s*"([^"]+)"', html_content)
        sesskey = sesskey_match.group(1) if sesskey_match else None
    
    draft_input = soup.find('input', {'name': 'files_filemanager'})
    if not draft_input:
        draft_input = soup.find('input', id=re.compile(r'id_files_filemanager'))
    draftitemid = draft_input.get('value') if draft_input else None
    if not draftitemid:
        itemid_match = re.search(r'itemid=(\d+)', html_content)
        if not itemid_match:
            itemid_match = re.search(r'"itemid"\s*:\s*(\d+)', html_content)
        draftitemid = itemid_match.group(1) if itemid_match else None
    
    client_id_match = re.search(r'id="filemanager-([a-zA-Z0-9]+)"', html_content)
    client_id = client_id_match.group(1) if client_id_match else None
    if not client_id:
        client_id_match = re.search(r'"client_id"\s*:\s*"([^"]+)"', html_content)
        client_id = client_id_match.group(1) if client_id_match else None
        
    return sesskey, draftitemid, client_id, soup

def process_workflow():
    processed_timestamps = {}
    current_sleep = 30  # Start by checking every 30 seconds
    
    while True:
        # Recreate session to force Moodle to generate a new draft area with updated files
        session = requests.Session()
        print(f"Checking for files at {FILES_URL}...")
        response = session.get(FILES_URL)
        
        # Moodle might return 303 Redirect or 200 OK with the login form
        if "name=\"logintoken\"" in response.text or response.status_code == 303:
            print("Authentication required. Logging in...")
            
            # Fetch the login page to get a fresh logintoken and session cookie
            login_page = session.get(LOGIN_URL)
            logintoken = get_login_token(login_page.text)
            
            if not logintoken:
                print("Could not find logintoken on the login page. Retrying in 60s...")
                time.sleep(60)
                continue

            login_data = {
                'username': USERNAME,
                'password': PASSWORD,
                'logintoken': logintoken,
                'anchor': ''
            }
            
            # Submit the login form
            session.post(LOGIN_URL, data=login_data)
            
            # Verify login by attempting to fetch the files page again
            response = session.get(FILES_URL)
            
            if "name=\"logintoken\"" in response.text:
                print("Login failed. Please check your credentials.")
                time.sleep(current_sleep)
                continue

        # If we successfully accessed the files page
        if response.status_code == 200 and "name=\"logintoken\"" not in response.text:
            sesskey, draftitemid, client_id, soup = extract_filemanager_params(response.text)
            
            if not all([sesskey, draftitemid, client_id]):
                print(f"Failed to extract parameters: sesskey={sesskey}, draftitemid={draftitemid}, client_id={client_id}")
                time.sleep(current_sleep)
                continue
                
            # List files in draft area
            list_url = f"{MOODLE_BASE_URL}/repository/draftfiles_ajax.php?action=list"
            list_data = {
                'sesskey': sesskey,
                'client_id': client_id,
                'filepath': '/',
                'itemid': draftitemid
            }
            
            list_response = session.post(list_url, data=list_data)
            try:
                files_list = list_response.json()
            except Exception as e:
                print(f"Error decoding draft files list: {e}")
                time.sleep(current_sleep)
                continue
                
            files_to_process = []
            for f_info in files_list.get('list', []):
                filename = f_info.get('filename', '')
                # Process any .txt file that is NOT an answer file
                if filename.endswith('.txt') and not filename.startswith('answers_') and filename != 'answers.txt':
                    questions_datemodified = f_info.get('datemodified', 0)
                    last_processed = processed_timestamps.get(filename, 0)
                    if questions_datemodified > last_processed:
                        files_to_process.append(f_info)
                        
            if not files_to_process:
                print(f"No new or updated .txt question files found. Waiting {current_sleep} seconds...")
                time.sleep(current_sleep)
                continue
                
            uploads_successful = 0
            
            for f_info in files_to_process:
                filename = f_info['filename']
                questions_url = f_info['url']
                questions_datemodified = f_info['datemodified']
                
                print(f"Downloading {filename} from {questions_url}...")
                file_response = session.get(questions_url)
                
                if file_response.status_code == 200:
                    print(f"File {filename} downloaded successfully!")
                    questions = file_response.text
                    print(f"Questions:\n{questions}\n")
                    
                    print(f"Querying AI Studio for {filename}...")
                    try:
                        # Call the AI API
                        ai_response = model.generate_content(questions)
                        answers = ai_response.text
                        print("Answers generated successfully.")
                        
                        if filename == 'questions.txt':
                            answers_filename = 'answers.txt'
                        else:
                            answers_filename = f"answers_{filename}"
                            
                        with open(answers_filename, "w", encoding="utf-8") as f:
                            f.write(answers)
                            
                        print(f"Saved to {answers_filename}.")
                        
                        # --- UPLOAD PHASE ---
                        print(f"Uploading {answers_filename} to Moodle via Web Scraping...")
                        
                        # Based on the user's network trace, the "Upload a file" repository ID is 4 on this server
                        repo_id = "4"
                        
                        # 3. Upload File to Draft Area
                        upload_url = f"{MOODLE_BASE_URL}/repository/repository_ajax.php?action=upload"
                        upload_data = {
                            'title': answers_filename,
                            'author': USERNAME,
                            'itemid': draftitemid,
                            'repo_id': repo_id,
                            'env': 'filemanager',
                            'sesskey': sesskey,
                            'client_id': client_id,
                            'savepath': '/'
                        }
                        with open(answers_filename, 'rb') as f:
                            files = {'repo_upload_file': f} # Moodle expects this field name
                            
                            try:
                                upload_res = session.post(upload_url, data=upload_data, files=files).json()
                            except Exception as e:
                                print(f"JSON Decode Error (Upload response might not be JSON): {e}")
                                upload_res = {'error': True}
                            
                        if 'error' in upload_res:
                            print(f"Failed to upload {answers_filename} to draft area: {upload_res}")
                        else:
                            print(f"File {answers_filename} successfully uploaded to draft area.")
                            uploads_successful += 1
                            # Update timestamp logic here so we don't process it again if it succeeded
                            processed_timestamps[filename] = questions_datemodified
                            
                    except Exception as e:
                        print(f"An error occurred while processing {filename} (AI or Upload): {e}")
                else:
                    print(f"Failed to download {filename}. Status Code: {file_response.status_code}")
                    
            # 4. Save Draft Area to Private Files if we uploaded anything
            if uploads_successful > 0:
                print(f"Saving {uploads_successful} new file(s) to private files...")
                save_data = {}
                mform = soup.find('form', {'class': 'mform'})
                if mform:
                    for hidden in mform.find_all('input', type='hidden'):
                        name = hidden.get('name')
                        value = hidden.get('value', '')
                        if name:
                            save_data[name] = value
                
                save_data['sesskey'] = sesskey
                save_data['files_filemanager'] = draftitemid
                save_data['submitbutton'] = 'Save changes'
                
                save_res = session.post(FILES_URL, data=save_data)
                
                if save_res.status_code == 200:
                    print("Upload complete. Saved all new files to private files.")
                    # SUCCESSFUL UPLOAD: Change polling interval to 15 minutes!
                    current_sleep = 900
                    print("Switching to 15-minute polling interval...")
                else:
                    print(f"Failed to save to private files. Status: {save_res.status_code}")

            print(f"Waiting {current_sleep} seconds before next cycle...")
            time.sleep(current_sleep)
        else:
            print(f"Unexpected status or content on files page. Status Code: {response.status_code}")
            time.sleep(current_sleep)

if __name__ == "__main__":
    # Session is recreated per cycle to avoid sticky draft areas
    process_workflow()
