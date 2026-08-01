import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import os
import traceback
from google import genai
from bs4 import BeautifulSoup
import json

# --- Configuration ---
MOODLE_BASE_URL = "http://103.117.208.18/moodle"
FILES_URL = f"{MOODLE_BASE_URL}/user/files.php"
LOGIN_URL = f"{MOODLE_BASE_URL}/login/index.php"

USERNAME = os.getenv("MOODLE_USERNAME")
PASSWORD = os.getenv("MOODLE_PASSWORD")
AI_API_KEY = os.getenv("AI_API_KEY")

if not USERNAME or not PASSWORD or not AI_API_KEY:
    print("Error: MOODLE_USERNAME, MOODLE_PASSWORD, and AI_API_KEY environment variables must be set.")
    print(f"  MOODLE_USERNAME set: {bool(USERNAME)}")
    print(f"  MOODLE_PASSWORD set: {bool(PASSWORD)}")
    print(f"  AI_API_KEY set: {bool(AI_API_KEY)}")
    exit(1)

# --- AI Provider Setup ---
gemini_client = genai.Client(api_key=AI_API_KEY)

# Fallback provider API keys (all optional)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize fallback clients
groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    except ImportError:
        print("WARNING: groq package not installed. pip install groq")
    except Exception as e:
        print(f"WARNING: Could not init Groq: {e}")

cerebras_client = None
if CEREBRAS_API_KEY:
    try:
        from cerebras.cloud.sdk import Cerebras
        cerebras_client = Cerebras(api_key=CEREBRAS_API_KEY)
    except ImportError:
        print("WARNING: cerebras_cloud_sdk not installed. pip install cerebras_cloud_sdk")
    except Exception as e:
        print(f"WARNING: Could not init Cerebras: {e}")

openrouter_client = None
if OPENROUTER_API_KEY:
    try:
        from openai import OpenAI
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    except ImportError:
        print("WARNING: openai package not installed. pip install openai")
    except Exception as e:
        print(f"WARNING: Could not init OpenRouter: {e}")

# --- Ordered Fallback Chain ---
# Each entry: (provider, model_id)
# Tried in EXACTLY this order. On ANY error, immediately skip to next.
FALLBACK_CHAIN = [
    ("gemini",     "gemini-2.5-flash"),                          # 1. Gemini 2.5 Flash
    ("openrouter", "qwen/qwen3-coder"),                         # 2. Qwen 3 Coder (480B MoE)
    ("groq",       "qwen/qwen3-32b"),                           # 3. Qwen 3 32B
    ("groq",       "openai/gpt-oss-120b"),                      # 4. GPT-OSS 120B
    ("groq",       "moonshotai/kimi-k2-instruct-0905"),         # 5. Kimi K2
    ("groq",       "llama-3.3-70b-versatile"),                  # 6. Llama 3.3 70B Versatile
    ("groq",       "meta-llama/llama-4-scout-17b-16e-instruct"),# 7. Llama 4 Scout
    ("openrouter", "deepseek/deepseek-v4-flash"),               # 8. DeepSeek V4 Flash
    ("gemini",     "gemini-3.5-flash-lite"),                    # 9. Gemini 3.5 Flash Lite
    ("openrouter", "mistralai/mistral-small"),                  # 10. Mistral Small
]

# Print startup summary
enabled = []
if groq_client: enabled.append("Groq")
if cerebras_client: enabled.append("Cerebras")
if openrouter_client: enabled.append("OpenRouter")
print(f"AI fallback chain: {len(FALLBACK_CHAIN)} models | Providers: Gemini{', ' + ', '.join(enabled) if enabled else ' only'}")
for i, (prov, model) in enumerate(FALLBACK_CHAIN, 1):
    print(f"  {i:>2}. [{prov}] {model}")

# System prompt: write like a beginner Java programmer
SYSTEM_PROMPT = (
    "You are a college student who is learning Java programming. "
    "Write code like a beginner - it doesn't have to be the cleanest or most optimized, "
    "but it should work correctly and get the job done. "
    "Use simple variable names, basic loops, and straightforward logic. "
    "Always include all necessary import statements at the top. "
    "For GUI programs, use javax.swing (JFrame, JPanel, JButton, etc.) and java.awt. "
    "Don't use advanced design patterns or lambda expressions unless absolutely needed. "
    "Keep it simple and functional - like a student who just learned the topic would write it. "
    "If the question is not about Java, just answer it normally in a simple and direct way."
)


def _call_gemini(model, prompt):
    """Call Gemini API. Returns answer text or raises."""
    resp = gemini_client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )
    return resp.text


def _call_openai_compat(client, model, prompt):
    """Call any OpenAI-compatible API (Groq, Cerebras, OpenRouter). Returns answer text or raises."""
    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
    )
    return resp.choices[0].message.content


def call_ai(prompt):
    """Try each model in FALLBACK_CHAIN in order. On ANY error, immediately try next.
    Returns (answer_text, provider/model) or raises if ALL fail."""
    
    errors = []
    
    for provider, model in FALLBACK_CHAIN:
        # Pick the right client
        if provider == "gemini":
            client_obj = gemini_client  # always available
        elif provider == "groq":
            client_obj = groq_client
        elif provider == "cerebras":
            client_obj = cerebras_client
        elif provider == "openrouter":
            client_obj = openrouter_client
        else:
            continue
        
        # Skip if provider not configured
        if client_obj is None:
            print(f"  [{provider}/{model}] Skipped (no API key)")
            continue
        
        try:
            print(f"  [{provider}/{model}] Trying...")
            if provider == "gemini":
                answer = _call_gemini(model, prompt)
            else:
                answer = _call_openai_compat(client_obj, model, prompt)
            
            if answer and answer.strip():
                print(f"  [{provider}/{model}] ✓ Success!")
                return answer, f"{provider}/{model}"
            else:
                print(f"  [{provider}/{model}] Empty response, trying next...")
        except Exception as e:
            print(f"  [{provider}/{model}] ✗ Failed: {e}")
            errors.append(f"{provider}/{model}: {e}")
            continue  # Immediately try next — no waiting
    
    raise Exception(f"ALL {len(FALLBACK_CHAIN)} models failed. Errors: {'; '.join(errors[-3:])}")

def get_login_token(html_content):
    """Extracts the Moodle logintoken from the login page HTML."""
    match = re.search(r'name="logintoken" value="([^"]+)"', html_content)
    if match:
        return match.group(1)
    return None

def extract_balanced_json(text, start_pos):
    """Extract a balanced JSON object starting from start_pos (which should point to '{')."""
    if start_pos >= len(text) or text[start_pos] != '{':
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start_pos, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start_pos:i+1]
    return None

def extract_filemanager_params(html_content):
    """Extracts required tokens from the Moodle files page."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # --- Extract sesskey ---
    sesskey_input = soup.find('input', {'name': 'sesskey'})
    sesskey = sesskey_input.get('value') if sesskey_input else None
    if not sesskey:
        sesskey_match = re.search(r'"sesskey"\s*:\s*"([^"]+)"', html_content)
        sesskey = sesskey_match.group(1) if sesskey_match else None
    
    # --- Extract filemanager init data from JavaScript ---
    # Moodle embeds the filemanager config (with itemid, client_id, and existing files)
    # in a Y.use / M.form_filemanager.init call in the page's JavaScript.
    fm_init_data = None
    
    # Pattern 1: M.form_filemanager.init(Y, {...});
    fm_match = re.search(r'M\.form_filemanager\.init\s*\(\s*Y\s*,\s*', html_content)
    if fm_match:
        json_str = extract_balanced_json(html_content, fm_match.end())
        if json_str:
            try:
                fm_init_data = json.loads(json_str)
                print(f"DEBUG: Found M.form_filemanager.init data with keys: {list(fm_init_data.keys())}")
            except json.JSONDecodeError as e:
                print(f"DEBUG: Failed to parse filemanager init JSON: {e}")
                print(f"DEBUG: Raw JSON string (first 500 chars): {json_str[:500]}")
    
    # Pattern 2: initializer_for form_filemanager in require() call  
    if not fm_init_data:
        fm_match2 = re.search(r'"form_filemanager"\s*,\s*', html_content)
        if fm_match2:
            json_str2 = extract_balanced_json(html_content, fm_match2.end())
            if json_str2:
                try:
                    fm_init_data = json.loads(json_str2)
                    print(f"DEBUG: Found form_filemanager data (pattern 2) with keys: {list(fm_init_data.keys())}")
                except json.JSONDecodeError:
                    pass
    
    # Pattern 3: Look for any JSON block containing both "itemid" and "client_id" near "filemanager"
    if not fm_init_data:
        for fm_match3 in re.finditer(r'filemanager', html_content, re.IGNORECASE):
            search_start = fm_match3.start()
            brace_pos = html_content.find('{', search_start)
            if brace_pos != -1 and brace_pos - search_start < 300:
                json_str3 = extract_balanced_json(html_content, brace_pos)
                if json_str3 and '"itemid"' in json_str3 and '"client_id"' in json_str3:
                    try:
                        fm_init_data = json.loads(json_str3)
                        print(f"DEBUG: Found filemanager data (pattern 3) with keys: {list(fm_init_data.keys())}")
                        break
                    except json.JSONDecodeError:
                        continue
    
    if not fm_init_data:
        # Dump HTML context around 'filemanager' for debugging
        fm_pos = html_content.lower().find('filemanager')
        if fm_pos != -1:
            start = max(0, fm_pos - 200)
            end = min(len(html_content), fm_pos + 2000)
            print(f"DEBUG: No filemanager init data found. HTML around 'filemanager' (pos {fm_pos}):")
            print(html_content[start:end])
        else:
            print("DEBUG: 'filemanager' not found anywhere in the page HTML!")
    
    # --- Extract draftitemid ---
    draftitemid = None
    if fm_init_data:
        draftitemid = str(fm_init_data.get('itemid', '')) or None
        print(f"DEBUG: itemid from JS init data: {draftitemid}")
    
    if not draftitemid:
        draft_input = soup.find('input', {'name': 'files_filemanager'})
        if not draft_input:
            draft_input = soup.find('input', id=re.compile(r'id_files_filemanager'))
        draftitemid = draft_input.get('value') if draft_input else None
        print(f"DEBUG: itemid from HTML input: {draftitemid}")
    
    if not draftitemid:
        # Try broader regex patterns
        itemid_match = re.search(r'"itemid"\s*:\s*(\d+)', html_content)
        if not itemid_match:
            itemid_match = re.search(r'itemid=(\d+)', html_content)
        draftitemid = itemid_match.group(1) if itemid_match else None
        print(f"DEBUG: itemid from regex fallback: {draftitemid}")
    
    # --- Extract client_id ---
    client_id = None
    if fm_init_data:
        client_id = fm_init_data.get('client_id')
        print(f"DEBUG: client_id from JS init data: {client_id}")
    
    if not client_id:
        client_id_match = re.search(r'id="filemanager-([a-zA-Z0-9]+)"', html_content)
        client_id = client_id_match.group(1) if client_id_match else None
        if not client_id:
            client_id_match = re.search(r'"client_id"\s*:\s*"([^"]+)"', html_content)
            client_id = client_id_match.group(1) if client_id_match else None
        print(f"DEBUG: client_id from HTML/regex: {client_id}")
    
    # --- Extract files already listed in the JS init data (fallback) ---
    js_files = []
    if fm_init_data and 'list' in fm_init_data:
        js_files = fm_init_data['list']
        # The list might be a dict (keyed by filename) or a list
        if isinstance(js_files, dict):
            js_files = list(js_files.values())
        print(f"DEBUG: Found {len(js_files)} files embedded in JS init data")
        for jf in js_files:
            print(f"  JS file: {jf.get('filename', 'unknown')}")
    
    return sesskey, draftitemid, client_id, soup, js_files

def process_workflow():
    processed_timestamps = {}
    current_sleep = 30  # Start by checking every 30 seconds
    
    print(f"MoodAI Worker started.")
    print(f"Target: {MOODLE_BASE_URL}")
    print(f"Username: {USERNAME}")
    print(f"Password set: {bool(PASSWORD)}")
    print(f"API Key set: {bool(AI_API_KEY)}")
    print(f"Groq fallback: {'enabled' if groq_client else 'disabled'}")
    print(f"Gemini model: {GEMINI_MODEL}")
    
    while True:
      try:
        # Recreate session to force Moodle to generate a new draft area with updated files
        session = requests.Session()
        # Add retry logic for transient connection errors
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        print(f"Checking for files at {FILES_URL}...")
        try:
            response = session.get(FILES_URL, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Could not connect to Moodle server: {e}")
            time.sleep(current_sleep)
            continue
        
        # Moodle might return 303 Redirect or 200 OK with the login form
        if "name=\"logintoken\"" in response.text or response.status_code == 303:
            print("Authentication required. Logging in...")
            
            try:
                # Fetch the login page to get a fresh logintoken and session cookie
                login_page = session.get(LOGIN_URL, timeout=30)
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
                login_response = session.post(LOGIN_URL, data=login_data, timeout=30)
                print(f"Login POST status: {login_response.status_code}")
                print(f"Login POST redirected to: {login_response.url}")
                
                # Check for error messages in the login response
                if 'loginerrors' in login_response.text or 'Invalid login' in login_response.text:
                    print("ERROR: Moodle returned login error - invalid credentials!")
                    time.sleep(current_sleep)
                    continue
                
                # Verify login by attempting to fetch the files page again
                response = session.get(FILES_URL, timeout=30)
                print(f"Files page status after login: {response.status_code}")
                
                # Try to extract logged-in username from page
                logged_in_match = re.search(r'loggedinuser.*?>(.+?)<', response.text)
                if logged_in_match:
                    print(f"Logged in as: {logged_in_match.group(1).strip()}")
                
                if "name=\"logintoken\"" in response.text:
                    print("Login failed. Still seeing login form after POST.")
                    print("Check that MOODLE_USERNAME and MOODLE_PASSWORD secrets are correct.")
                    time.sleep(current_sleep)
                    continue
            except requests.exceptions.RequestException as e:
                print(f"ERROR: Connection error during login: {e}")
                print(f"Retrying in {current_sleep} seconds...")
                time.sleep(current_sleep)
                continue

        # If we successfully accessed the files page
        if response.status_code == 200 and "name=\"logintoken\"" not in response.text:
            sesskey, draftitemid, client_id, soup, js_files = extract_filemanager_params(response.text)
            
            print(f"DEBUG: Extracted params -> sesskey={sesskey}, draftitemid={draftitemid}, client_id={client_id}")
            
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
            
            try:
                list_response = session.post(list_url, data=list_data, timeout=30)
            except requests.exceptions.RequestException as e:
                print(f"ERROR: Draft files AJAX request failed: {e}")
                time.sleep(current_sleep)
                continue
            print(f"DEBUG: Draft files AJAX status: {list_response.status_code}")
            print(f"DEBUG: Draft files AJAX response (first 1000 chars): {list_response.text[:1000]}")
            try:
                files_list = list_response.json()
            except Exception as e:
                print(f"Error decoding draft files list: {e}")
                time.sleep(current_sleep)
                continue
                
            files_to_process = []
            all_files = files_list.get('list', [])
            
            # If AJAX returned no files but we got files from the JS init data, use those
            if not all_files and js_files:
                print(f"AJAX returned 0 files but JS init data has {len(js_files)} files. Using JS data.")
                all_files = js_files
            
            print(f"Total files in Moodle private files: {len(all_files)}")
            for f_info in all_files:
                filename = f_info.get('filename', '')
                print(f"  Found: {filename} (modified: {f_info.get('datemodified', 'N/A')})")
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
                try:
                    file_response = session.get(questions_url, timeout=30)
                except requests.exceptions.RequestException as e:
                    print(f"ERROR: Failed to download {filename}: {e}")
                    continue
                
                if file_response.status_code == 200:
                    print(f"File {filename} downloaded successfully!")
                    questions = file_response.text
                    print(f"Questions:\n{questions}\n")
                    
                    print(f"Querying AI for {filename}...")
                    try:
                        # Call AI with automatic Gemini -> Groq fallback
                        answers, provider = call_ai(questions)
                        print(f"Answers generated successfully via {provider}.")
                        
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
                                upload_res = session.post(upload_url, data=upload_data, files=files, timeout=60).json()
                            except requests.exceptions.RequestException as e:
                                print(f"ERROR: Upload request failed for {answers_filename}: {e}")
                                upload_res = {'error': True}
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
                
                try:
                    save_res = session.post(FILES_URL, data=save_data, timeout=30)
                    
                    if save_res.status_code == 200:
                        print("Upload complete. Saved all new files to private files.")
                        # SUCCESSFUL UPLOAD: Change polling interval to 15 minutes!
                        current_sleep = 900
                        print("Switching to 15-minute polling interval...")
                    else:
                        print(f"Failed to save to private files. Status: {save_res.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"ERROR: Failed to save to private files: {e}")

            print(f"Waiting {current_sleep} seconds before next cycle...")
            time.sleep(current_sleep)
        else:
            print(f"Unexpected status or content on files page. Status Code: {response.status_code}")
            time.sleep(current_sleep)

      except Exception as e:
        # Top-level safety net: catch ANY unexpected error so the worker never dies
        print(f"CRITICAL: Unexpected error in main loop: {e}")
        traceback.print_exc()
        print(f"Recovering... waiting {current_sleep} seconds before next cycle.")
        time.sleep(current_sleep)

if __name__ == "__main__":
    # Session is recreated per cycle to avoid sticky draft areas
    process_workflow()
