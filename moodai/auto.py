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

from project_agent import build_project

# --- Configuration ---
MOODLE_BASE_URL = "http://103.117.208.18/moodle"
FILES_URL = f"{MOODLE_BASE_URL}/user/files.php"
LOGIN_URL = f"{MOODLE_BASE_URL}/login/index.php"
POLL_INTERVAL = 5  # seconds between checks
COOLDOWN_SECONDS = 10  # skip a model this long after it errors (rate limits etc.)
RETRY_AFTER_FAIL = 120  # don't retry a failed question file for this long

USERNAME = os.getenv("MOODLE_USERNAME")
PASSWORD = os.getenv("MOODLE_PASSWORD")
AI_API_KEY = os.getenv("AI_API_KEY")

# MAX_RUNTIME_MIN unset -> run forever (local use); "0" -> one pass then exit;
# "150" -> keep polling for 150 minutes then exit (GitHub Actions).
_raw_runtime = os.getenv("MAX_RUNTIME_MIN")
DEADLINE = float("inf") if _raw_runtime in (None, "") else time.time() + float(_raw_runtime) * 60

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

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openrouter_client = None
if OPENROUTER_API_KEY:
    try:
        from openai import OpenAI
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    except (ImportError, Exception) as e:
        print(f"WARNING: OpenRouter unavailable: {e}")

# Put your strongest models first - planner / fixer / coder all start from the top.
FALLBACK_CHAIN = [
    ("gemini",     "gemini-3.8-flash"),
    ("openrouter", "poolside/laguna-s-2.1:free"),
    ("openrouter", "nvidia/nemotron-3.5-super-vl:free"),
    ("groq",       "openai/gpt-oss-120b"),
    ("groq",       "qwen/qwen3.8-27b"),
    ("openrouter", "google/gemma-4-26b-a4b-it:free"),
    ("gemini",     "gemini-3.6-flash"),
    ("openrouter", "cohere/north-mini-code:free"),
    ("groq",       "openai/gpt-oss-20b"),
    ("gemini",     "gemini-3.5-flash-lite"),
]

# The reviewer starts from a different model than the coder, so it doesn't share its blind spots.
ROLE_CHAINS = {
    "planner":  FALLBACK_CHAIN,
    "coder":    FALLBACK_CHAIN,
    "fixer":    FALLBACK_CHAIN,
    "reviewer": FALLBACK_CHAIN[1:] + FALLBACK_CHAIN[:1],
}

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

print(f"MoodAI | {len(FALLBACK_CHAIN)} models | Groq: {'on' if groq_client else 'off'} | OpenRouter: {'on' if openrouter_client else 'off'}")

cooldown = {}  # (provider, model) -> time until which the model is skipped


# --- AI Call with Fallback ---
def _ask(provider, model, prompt, system):
    client = {"gemini": gemini_client, "groq": groq_client, "openrouter": openrouter_client}.get(provider)
    if client is None:
        return None
    if provider == "gemini":
        resp = client.models.generate_content(
            model=model, contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system, temperature=0.2,
            ),
        )
        return resp.text
    # groq and openrouter are both OpenAI-compatible
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def call_ai(prompt, system=SYSTEM_PROMPT, role="coder"):
    """Try each model for this role in order. Returns (answer, model_name) or raises.
    Models that errored recently are tried last instead of being skipped entirely."""
    chain = ROLE_CHAINS.get(role, FALLBACK_CHAIN)
    now = time.time()
    ready = [k for k in chain if cooldown.get(k, 0) <= now]
    waiting = [k for k in chain if cooldown.get(k, 0) > now]
    if waiting:
        skipped = ', '.join(f"{p}/{m}" for p, m in waiting)
        print(f"  AI[{role}]: skipping (cooldown): {skipped}")
    ordered = ready + waiting
    errors = []
    for provider, model in ordered:
        try:
            answer = _ask(provider, model, prompt, system)
            if answer and answer.strip():
                print(f"  AI[{role}]: {provider}/{model} OK")
                return answer, f"{provider}/{model}"
        except Exception as e:
            cooldown[(provider, model)] = time.time() + COOLDOWN_SECONDS
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
    Uses only regex - no BeautifulSoup needed."""

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

    # Hidden form fields (for the save POST) - regex instead of BeautifulSoup
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
            print("  Login failed - still on login page")
            return None

        # Deliberately not printing the account name: Actions logs can be public.
        print("  Logged in")
        return files_resp
    except requests.exceptions.RequestException as e:
        print(f"  Connection error: {e}")
        return None


def text_answer_name(name):
    return 'answers.txt' if name == 'questions.txt' else f"answers_{name}"


def project_name_from(name):
    """project_bank_system.txt -> BankSystem  (deterministic, so a finished question is never redone)"""
    base = re.sub(r'^project_', '', os.path.splitext(name)[0])
    parts = [p for p in re.split(r'[^A-Za-z0-9]+', base) if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) or "JavaProject"


def answer_names(name):
    """Every file name that counts as 'this question is already answered'."""
    if name.startswith('project_'):
        return [f"{project_name_from(name)}.zip", text_answer_name(name)]  # text = fallback answer
    return [text_answer_name(name)]


def make_answer(name, questions):
    """Returns (payload_bytes, upload_name, mimetype). project_*.txt -> zipped NetBeans project."""
    if name.startswith('project_'):
        pname = project_name_from(name)
        print(f"  {name}: PROJECT MODE - building NetBeans project '{pname}'")
        try:
            data, report = build_project(
                pname, questions, lambda p, s, r: call_ai(p, s, r)[0], log=print)
            if not report["ok"]:
                print(f"  {name}: WARNING - not fully fixed, see BUILD_NOTES.txt inside the zip")
            elif not report["verified"]:
                print(f"  {name}: note - needs extra libraries, could not be compiled automatically")
            return data, f"{pname}.zip", 'application/zip'
        except Exception as e:
            print(f"  {name}: project build failed ({e}); falling back to a text answer")
    answers, model = call_ai(questions)
    print(f"  {name}: answered via {model}")
    return answers.encode('utf-8'), text_answer_name(name), 'text/plain'


def upload_file(session, filename, filedata, mimetype, sesskey, itemid, client_id):
    """Upload a file to the Moodle draft area. Returns True on success."""
    try:
        upload = session.post(
            f"{MOODLE_BASE_URL}/repository/repository_ajax.php?action=upload",
            data={
                'title': filename, 'author': USERNAME,
                'itemid': itemid, 'repo_id': '4',
                'env': 'filemanager', 'sesskey': sesskey,
                'client_id': client_id, 'savepath': '/',
                'overwrite': 'true',  # replace an older answer if the question was edited
            },
            files={'repo_upload_file': (filename, io.BytesIO(filedata), mimetype)},
            timeout=120,
        ).json()
    except Exception as e:
        print(f"  {filename}: upload failed: {e}")
        return False
    if 'error' in upload:
        print(f"  {filename}: upload error: {upload}")
        return False
    print(f"  {filename}: uploaded OK")
    return True


# --- Main Loop ---
def wait():
    """Sleep one poll interval. Returns False when the time budget is used up."""
    if time.time() + POLL_INTERVAL > DEADLINE:
        return False
    time.sleep(POLL_INTERVAL)
    return True


def main():
    processed = {}    # filename -> datemodified we already handled in THIS run
    retry_after = {}  # filename -> time before which we won't retry a failure

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "OPTIONS"],
    ))
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    logged_in = False

    print(f"Polling {MOODLE_BASE_URL} every {POLL_INTERVAL}s")

    while True:
        try:
            # 1. Fetch files page (doubles as session check)
            try:
                resp = session.get(FILES_URL, timeout=30)
            except requests.exceptions.RequestException as e:
                print(f"Connection error: {e}")
                if not wait():
                    break
                continue

            # 2. Login only if Moodle shows login form
            if 'name="logintoken"' in resp.text or resp.status_code == 303:
                print("Session expired, re-authenticating..." if logged_in else "Logging in...")
                resp = moodle_login(session)
                if not resp:
                    logged_in = False
                    if not wait():
                        break
                    continue
                logged_in = True
            elif not logged_in:
                logged_in = True

            # 3. Extract page parameters
            sesskey, itemid, client_id, js_files, hidden = extract_page_params(resp.text)
            if not all([sesskey, itemid, client_id]):
                print("Missing page params (sesskey/itemid/client_id)")
                if not wait():
                    break
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

            # 5. Stateless "is there work to do?" - the answer files in Moodle ARE the state,
            #    so a fresh GitHub Actions run never redoes finished questions.
            existing = {f.get('filename'): int(f.get('datemodified') or 0) for f in all_files}
            now = time.time()
            new_files = []
            for f in all_files:
                name = f.get('filename', '')
                modified = int(f.get('datemodified') or 0)
                if not name.endswith('.txt') or name.startswith('answers_') or name == 'answers.txt':
                    continue
                if modified <= processed.get(name, 0) or retry_after.get(name, 0) > now:
                    continue
                if any(existing.get(n, 0) >= modified for n in answer_names(name)):
                    continue
                new_files.append(f)

            if not new_files:
                if not wait():
                    break
                continue

            # 6. Process each question file
            print(f"Found {len(new_files)} new question file(s)")
            uploads = 0

            for f in new_files:
                name = f['filename']
                modified = int(f.get('datemodified') or 0)

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

                # Build the answer: project_*.txt -> NetBeans zip, anything else -> text answer
                try:
                    payload, ans_name, mime = make_answer(name, questions)
                except Exception as e:
                    print(f"  {name}: failed: {e}")
                    retry_after[name] = time.time() + RETRY_AFTER_FAIL
                    continue

                if upload_file(session, ans_name, payload, mime, sesskey, itemid, client_id):
                    uploads += 1
                    processed[name] = modified
                else:
                    retry_after[name] = time.time() + RETRY_AFTER_FAIL

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

        if not wait():
            break

    print("Time budget used up - exiting.")


if __name__ == "__main__":
    main()
