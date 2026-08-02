import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import os
import io
import traceback
from google import genai
import json

# --- Configuration ---
MOODLE_BASE_URL = "http://103.117.208.18/moodle"
FILES_URL = f"{MOODLE_BASE_URL}/user/files.php"
LOGIN_URL = f"{MOODLE_BASE_URL}/login/index.php"
POLL_INTERVAL = 5  # seconds between checks

USERNAME = os.getenv("MOODLE_USERNAME")
PASSWORD = os.getenv("MOODLE_PASSWORD")
AI_API_KEY = os.getenv("AI_API_KEY")

if not USERNAME or not PASSWORD or not AI_API_KEY:
    print("Error: MOODLE_USERNAME, MOODLE_PASSWORD, and AI_API_KEY must be set.")
    exit(1)

# --- AI Setup ---
gemini_client = genai.Client(api_key=AI_API_KEY)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    except (ImportError, Exception) as e:
        print(f"WARNING: Groq unavailable: {e}")

FALLBACK_CHAIN = [
    ("gemini", "gemini-3.5-flash"),
    ("groq",   "openai/gpt-oss-120b"),
    ("groq",   "qwen/qwen3.6-27b"),
    ("groq",   "llama-3.3-70b-versatile"),
    ("groq",   "openai/gpt-oss-20b"),
    ("groq",   "llama-3.1-8b-instant"),
    ("gemini", "gemini-3.5-flash-lite"),
]

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

print(f"MoodAI | {len(FALLBACK_CHAIN)} models | Groq: {'on' if groq_client else 'off'}")


# --- AI Call with Fallback ---
def call_ai(prompt):
    """Try each model in order. Returns (answer, model_name) or raises."""
    errors = []
    for provider, model in FALLBACK_CHAIN:
        client = gemini_client if provider == "gemini" else groq_client
        if client is None:
            continue
        try:
            if provider == "gemini":
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT, temperature=0.2,
                    ),
                )
                answer = resp.text
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                answer = resp.choices[0].message.content

            if answer and answer.strip():
                print(f"  AI: {provider}/{model} OK")
                return answer, f"{provider}/{model}"
        except Exception as e:
            errors.append(f"{provider}/{model}: {e}")
    raise Exception(f"All models failed: {'; '.join(errors[-3:])}")


# --- Moodle Helpers ---
def _extract_json_block(text, pos):
    """Extract a balanced {...} JSON object starting at pos."""
    if pos >= len(text) or text[pos] != '{':
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(pos, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == '\\' and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[pos:i + 1]
    return None


def _regex_first(html, *patterns):
    """Try multiple regex patterns, return first match group(1) or None."""
    for p in patterns:
        m = re.search(p, html)
        if m:
            return m.group(1)
    return None


def extract_page_params(html):
    """Extract sesskey, draftitemid, client_id, file list, and hidden form fields.
    Uses only regex — no BeautifulSoup needed."""

    # sesskey
    sesskey = _regex_first(html,
        r'name="sesskey"\s+value="([^"]+)"',
        r'"sesskey"\s*:\s*"([^"]+)"',
    )

    # filemanager init JSON (embedded in page JS)
    fm = None
    for pattern in [r'M\.form_filemanager\.init\s*\(\s*Y\s*,\s*', r'"form_filemanager"\s*,\s*']:
        m = re.search(pattern, html)
        if m:
            raw = _extract_json_block(html, m.end())
            if raw:
                try:
                    fm = json.loads(raw)
                    break
                except json.JSONDecodeError:
                    pass

    # Fallback: any JSON near "filemanager" with itemid + client_id
    if not fm:
        for m in re.finditer(r'filemanager', html, re.IGNORECASE):
            brace = html.find('{', m.start())
            if brace != -1 and brace - m.start() < 300:
                raw = _extract_json_block(html, brace)
                if raw and '"itemid"' in raw and '"client_id"' in raw:
                    try:
                        fm = json.loads(raw)
                        break
                    except json.JSONDecodeError:
                        continue

    # draftitemid
    itemid = str(fm.get('itemid', '')) if fm else None
    if not itemid:
        itemid = _regex_first(html,
            r'name="files_filemanager"\s+value="(\d+)"',
            r'"itemid"\s*:\s*(\d+)',
            r'itemid=(\d+)',
        )

    # client_id
    client_id = fm.get('client_id') if fm else None
    if not client_id:
        client_id = _regex_first(html,
            r'id="filemanager-([a-zA-Z0-9]+)"',
            r'"client_id"\s*:\s*"([^"]+)"',
        )

    # Files embedded in JS init data
    js_files = []
    if fm and 'list' in fm:
        js_files = fm['list']
        if isinstance(js_files, dict):
            js_files = list(js_files.values())

    # Hidden form fields (for the save POST) — regex instead of BeautifulSoup
    hidden = {}
    for tag in re.finditer(r'<input[^>]+type=["\']hidden["\'][^>]*>', html):
        nm = re.search(r'name=["\']([^"\']+)["\']', tag.group(0))
        vl = re.search(r'value=["\']([^"\']*)["\']', tag.group(0))
        if nm:
            hidden[nm.group(1)] = vl.group(1) if vl else ''

    return sesskey, itemid, client_id, js_files, hidden


def moodle_login(session):
    """Perform Moodle login. Returns the files page response or None."""
    try:
        page = session.get(LOGIN_URL, timeout=30)
        token = _regex_first(page.text, r'name="logintoken" value="([^"]+)"')
        if not token:
            print("  No logintoken found")
            return None

        resp = session.post(LOGIN_URL, data={
            'username': USERNAME, 'password': PASSWORD,
            'logintoken': token, 'anchor': ''
        }, timeout=30)

        if 'loginerrors' in resp.text or 'Invalid login' in resp.text:
            print("  ERROR: Invalid credentials!")
            return None

        files_resp = session.get(FILES_URL, timeout=30)
        if 'name="logintoken"' in files_resp.text:
            print("  Login failed — still on login page")
            return None

        user = _regex_first(files_resp.text, r'loggedinuser[^>]*>([^<]+)<')
        if user:
            print(f"  Logged in as: {user.strip()}")
        return files_resp
    except requests.exceptions.RequestException as e:
        print(f"  Connection error: {e}")
        return None


# --- Main Loop ---
def main():
    processed = {}  # filename -> last datemodified we processed

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "OPTIONS"],
    ))
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    logged_in = False

    print(f"Polling {MOODLE_BASE_URL} every {POLL_INTERVAL}s as {USERNAME}")

    while True:
        try:
            # 1. Fetch files page (doubles as session check)
            try:
                resp = session.get(FILES_URL, timeout=30)
            except requests.exceptions.RequestException as e:
                print(f"Connection error: {e}")
                time.sleep(POLL_INTERVAL)
                continue

            # 2. Login only if Moodle shows login form
            if 'name="logintoken"' in resp.text or resp.status_code == 303:
                print("Session expired, re-authenticating..." if logged_in else "Logging in...")
                resp = moodle_login(session)
                if not resp:
                    logged_in = False
                    time.sleep(POLL_INTERVAL)
                    continue
                logged_in = True
            elif not logged_in:
                logged_in = True

            # 3. Extract page parameters
            sesskey, itemid, client_id, js_files, hidden = extract_page_params(resp.text)
            if not all([sesskey, itemid, client_id]):
                print(f"Missing params: sesskey={sesskey}, itemid={itemid}, client_id={client_id}")
                time.sleep(POLL_INTERVAL)
                continue

            # 4. List files via AJAX
            try:
                ajax = session.post(
                    f"{MOODLE_BASE_URL}/repository/draftfiles_ajax.php?action=list",
                    data={'sesskey': sesskey, 'client_id': client_id, 'filepath': '/', 'itemid': itemid},
                    timeout=30,
                )
                all_files = ajax.json().get('list', [])
            except Exception:
                all_files = []

            if not all_files and js_files:
                all_files = js_files

            # 5. Filter for new/updated question files
            new_files = [
                f for f in all_files
                if f.get('filename', '').endswith('.txt')
                and not f.get('filename', '').startswith('answers_')
                and f.get('filename') != 'answers.txt'
                and f.get('datemodified', 0) > processed.get(f.get('filename', ''), 0)
            ]

            if not new_files:
                time.sleep(POLL_INTERVAL)
                continue

            # 6. Process each question file
            print(f"Found {len(new_files)} new question file(s)")
            uploads = 0

            for f in new_files:
                name = f['filename']
                modified = f['datemodified']

                # Download
                try:
                    dl = session.get(f['url'], timeout=30)
                    if dl.status_code != 200:
                        print(f"  {name}: download failed ({dl.status_code})")
                        continue
                except requests.exceptions.RequestException as e:
                    print(f"  {name}: download error: {e}")
                    continue

                questions = dl.text
                print(f"  {name}: {len(questions)} chars")

                # Get AI answer
                try:
                    answers, model = call_ai(questions)
                    print(f"  {name}: answered via {model}")
                except Exception as e:
                    print(f"  {name}: AI failed: {e}")
                    continue

                # Upload directly from memory (no disk I/O)
                ans_name = 'answers.txt' if name == 'questions.txt' else f"answers_{name}"
                try:
                    upload = session.post(
                        f"{MOODLE_BASE_URL}/repository/repository_ajax.php?action=upload",
                        data={
                            'title': ans_name, 'author': USERNAME,
                            'itemid': itemid, 'repo_id': '4',
                            'env': 'filemanager', 'sesskey': sesskey,
                            'client_id': client_id, 'savepath': '/',
                        },
                        files={'repo_upload_file': (ans_name, io.BytesIO(answers.encode('utf-8')), 'text/plain')},
                        timeout=60,
                    ).json()
                except Exception as e:
                    print(f"  {ans_name}: upload failed: {e}")
                    continue

                if 'error' in upload:
                    print(f"  {ans_name}: upload error: {upload}")
                else:
                    print(f"  {ans_name}: uploaded OK")
                    uploads += 1
                    processed[name] = modified

            # 7. Save draft area to private files
            if uploads > 0:
                save_data = dict(hidden)
                save_data.update({
                    'sesskey': sesskey,
                    'files_filemanager': itemid,
                    'submitbutton': 'Save changes',
                })
                try:
                    save = session.post(FILES_URL, data=save_data, timeout=30)
                    status = "OK" if save.status_code == 200 else f"failed ({save.status_code})"
                    print(f"Saved {uploads} file(s) to private files: {status}")
                except requests.exceptions.RequestException as e:
                    print(f"Save error: {e}")

        except Exception as e:
            print(f"CRITICAL: {e}")
            traceback.print_exc()

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
