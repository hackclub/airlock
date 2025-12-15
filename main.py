import os
import json
import base64
import asyncio
import httpx
import markdown
import logging
import uuid
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from typing import List, Optional
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HC_AI_KEY = os.getenv("HACKCLUB_AI_KEY")
KASM_API_KEY = os.getenv("KASM_API_KEY")
KASM_API_SECRET = os.getenv("KASM_API_KEY_SECRET")
KASM_URL = os.getenv("KASM_SERVER_URL", "").rstrip('/')
KASM_IMAGE_ID = os.getenv("KASM_IMAGE_ID")

# Auth Configuration
HC_CLIENT_ID = os.getenv("HACKCLUB_CLIENT_ID")
HC_CLIENT_SECRET = os.getenv("HACKCLUB_CLIENT_SECRET")
APP_SECRET = os.getenv("APP_SECRET")

# Slack Configuration
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")
slack_client = WebClient(token=SLACK_BOT_TOKEN)

# Access Control
# Admins are loaded from ENV
ADMIN_USERS = os.getenv("ADMIN_USERS", "").split(",")
# Regular users are stored in JSON
USERS_FILE = "users.json"
ORGANIZATIONS_FILE = "organizations.json"

# Caches
slack_profile_cache = TTLCache(maxsize=1000, ttl=600)  # 10 minutes cache

# Models
class User(BaseModel):
    slack_id: str
    email: Optional[str] = None

class Organization(BaseModel):
    id: str
    name: str
    kasm_user_id: Optional[str] = None
    session_limit: int = 10
    admins: List[str] = []
    users: List[str] = []

class OrganizationUpdate(BaseModel):
    name: Optional[str] = None
    kasm_user_id: Optional[str] = None
    session_limit: Optional[int] = None

# Initialize FastAPI
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=APP_SECRET)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize OAuth
oauth = OAuth()
oauth.register(
    name='hackclub',
    client_id=HC_CLIENT_ID,
    client_secret=HC_CLIENT_SECRET,
    server_metadata_url='https://auth.hackclub.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid profile email slack_id'
    }
)

# --- Helper Functions ---

def load_users():
    if not os.path.exists(USERS_FILE):
        return []
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_organizations():
    if not os.path.exists(ORGANIZATIONS_FILE):
        return []
    try:
        with open(ORGANIZATIONS_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_organizations(orgs):
    with open(ORGANIZATIONS_FILE, 'w') as f:
        json.dump(orgs, f, indent=2)

def get_slack_profile(slack_id: str):
    if slack_id in slack_profile_cache:
        return slack_profile_cache[slack_id]
    
    try:
        response = slack_client.users_info(user=slack_id)
        if response["ok"]:
            user = response["user"]
            profile = user.get("profile", {})
            data = {
                "id": user["id"],
                "display_name": profile.get("display_name") or profile.get("real_name") or user["name"],
                "image": profile.get("image_48"),
                "image_192": profile.get("image_192")
            }
            slack_profile_cache[slack_id] = data
            return data
    except Exception as e:
        logger.error(f"Error fetching Slack profile for {slack_id}: {e}")
    
    return {"id": slack_id, "display_name": slack_id, "image": None}

def get_user_organization(slack_id):
    orgs = load_organizations()
    for org in orgs:
        if slack_id in org['admins'] or slack_id in org['users']:
            return org
    return None

def is_user_in_channel(user_id):
    try:
        # Initial call to get the first page of members
        response = slack_client.conversations_members(channel=SLACK_CHANNEL_ID, limit=1000)
        members = response["members"]
        
        while True:
            # Check if user is in the current batch of members
            if user_id in members:
                return True
            
            # Check if there is another page of members
            cursor = response.get("response_metadata", {}).get("next_cursor")
            
            if not cursor:
                # No more pages, user was not found
                break
            
            # Fetch the next page using the cursor
            response = slack_client.conversations_members(channel=SLACK_CHANNEL_ID, cursor=cursor, limit=1000)
            members = response["members"]
            
        return False

    except Exception as e:
        logger.error(f"Error checking channel members: {e}")
        return False

def is_admin(slack_id):
    return slack_id in ADMIN_USERS

def is_org_admin(slack_id, org_id):
    orgs = load_organizations()
    for org in orgs:
        if org['id'] == org_id:
            return slack_id in org['admins']
    return False

def is_any_org_admin(slack_id):
    orgs = load_organizations()
    for org in orgs:
        if slack_id in org['admins']:
            return True
    return False

def is_authorized(slack_id):
    if is_admin(slack_id):
        return True
    
    # Check if in an organization
    if get_user_organization(slack_id):
        return True

    # Check regular users list
    users = load_users()
    if any(u['slack_id'] == slack_id for u in users):
        return True
        
    return is_user_in_channel(slack_id)

def normalize_github_url(url):
    url = url.strip()
    # Basic protocol fix
    if not url.startswith("http"):
        url = "https://" + url
    
    # Check domain
    if "github.com" not in url:
        raise ValueError("URL must be a GitHub URL (github.com)")

    from urllib.parse import urlparse
    parsed = urlparse(url)
    
    path_parts = [p for p in parsed.path.split('/') if p]
    
    if len(path_parts) < 2:
        raise ValueError("Invalid GitHub URL format. Expected github.com/owner/repo")
        
    owner = path_parts[0]
    repo = path_parts[1]
    
    normalized = f"https://github.com/{owner}/{repo}"
    
    warning = None
    if len(path_parts) > 2:
        warning = f"URL contained extra path segments ('/{'/'.join(path_parts[2:])}'). Normalized to {normalized}."
        
    return normalized, warning

async def get_github_context_data(repo_url):
    if not "github.com/" in repo_url:
        raise ValueError("Invalid repo url")

    # Ensure protocol
    if not repo_url.startswith("http"):
        repo_url = f"https://{repo_url}"

    parts = repo_url.rstrip("/").split("/")
    # Find where github.com is
    try:
        gh_index = parts.index("github.com")
        if len(parts) < gh_index + 3:
             raise ValueError("Invalid repo url structure")
        owner = parts[gh_index + 1]
        repo = parts[gh_index + 2]
    except ValueError:
        # Fallback if github.com not found directly in split (unlikely if check passed)
        raise ValueError("Invalid repo url structure")

    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

    readme = ""
    files_list = []

    async with httpx.AsyncClient() as client:
        # First check if the repo exists and is accessible
        try:
            r = await client.get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers)
            if r.status_code == 404:
                raise ValueError("Repository not found (404). Please check the URL.")
            elif r.status_code != 200:
                raise ValueError(f"Could not access repository (Status: {r.status_code})")
        except httpx.RequestError as e:
            raise ValueError(f"Network error accessing GitHub: {e}")

        try:
            r = await client.get(f"https://api.github.com/repos/{owner}/{repo}/readme", headers=headers)
            if r.status_code == 200:
                readme = base64.b64decode(r.json()['content']).decode('utf-8')[:4000]
        except: pass

        try:
            r = await client.get(f"https://api.github.com/repos/{owner}/{repo}/contents", headers=headers)
            if r.status_code == 200:
                files_data = r.json()
                if not files_data:
                     raise ValueError("Repository is empty.")
                     
                # Limit to top 300 files to prevent context explosion
                files_data = files_data[:300]
                files_list = [f"{i['type']}: {i['path']} (size: {i['size']})" for i in files_data]
                if len(r.json()) > 300:
                    files_list.append("... (truncated)")
            elif r.status_code == 404:
                 # Should have been caught by the first check, but double check contents endpoint
                 raise ValueError("Repository contents not found (empty or invalid).")
            else:
                 # If we can't get contents, we can't analyze it properly
                 raise ValueError(f"Could not list repository contents (Status: {r.status_code})")

        except ValueError as e:
            raise e
        except Exception as e:
             raise ValueError(f"Error listing repository files: {e}")

    return {"readme": readme, "files": "\n".join(files_list), "name": repo, "owner": owner, "repo": repo}

async def fetch_github_file_content(owner, repo, path):
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"https://api.github.com/repos/{owner}/{repo}/contents/{path}", headers=headers)
            if r.status_code == 200:
                data = r.json()
                if 'content' in data:
                     return base64.b64decode(data['content']).decode('utf-8')
        except:
             pass
    return ""

async def pre_analyze_project(context):
    prompt = f"""
    Context:
    Repo Name: {context['name']}
    File Structure: {context['files']}
    README Content: {context['readme']}
    
    Task:
    Analyze the project structure and determine:
    1. Difficulty: "easy" or "hard".
       - Easy examples: Setting up some python dependencies, simple static site.
       - Hard examples: Setting up a complex website with a frontend and a backend and all of that.
    2. Tech Stack: A short string identifying the tech stack.
    3. Files to Read: A list of specific file paths (from the structure) that would provide critical context for writing an install script.
       - Select key files like package.json, requirements.txt, Dockerfile, main.py, etc. The system does not have docker so if the project requires docker, you should consider it hard and include the files docker uses on the list of files_to_read.
    
    Output Format (JSON only):
    {{
      "difficulty": "easy" | "hard",
      "tech_stack": "string",
      "files_to_read": ["path/to/file1", "path/to/file2"]
    }}
    """
    
    payload = {
        "model": "google/gemini-2.5-flash-lite-preview-09-2025",
        "messages": [{"role": "system", "content": prompt}],
        "temperature": 0.5
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post("https://ai.hackclub.com/proxy/v1/chat/completions",
                                  headers={"Authorization": f"Bearer {HC_AI_KEY}"}, json=payload)
            r.raise_for_status()
            content = r.json()['choices'][0]['message']['content']
            
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "")
            
            return json.loads(content)
    except Exception as e:
        logger.error(f"Pre-analysis failed: {e}")
        return {"difficulty": "easy", "tech_stack": "Unknown", "files_to_read": []}

async def analyze_with_ai_data(context, file_contents="", model_id="z-ai/glm-4.6"):
    # Determine retries based on model (easy/hard proxy)
    # easy (glm-4.6) -> 2 retries (total 3 attempts)
    # hard (gemini-3) -> 1 retry (total 2 attempts)
    max_retries = 2 if "glm" in model_id else 1
    
    prompt = f"""
    Context:
    Repo Name: {context['name']}
    File Structure: {context['files']}
    README Content: {context['readme']}
    Additional File Contents:
    {file_contents}

    System Environment:
    - OS: Ubuntu 22.04 (Jammy Jellyfish)
    - User: Sudo privileges available (no password required).
    - Python 3 is pre-installed.
    - The git repo is already cloned; the script runs from the repo root.
    - The system cannot run docker and has no systemd. If a project requires docker, you must run it manually.
    - The system has the following tooling installed:
    1. Core System & Build Tools
    Editors & Utilities: vim, nano, htop, jq, tree, curl, wget, git, unzip, zip, tar, gzip
    Build Essentials: build-essential, cmake, pkg-config, autoconf, automake, libtool
    Media: ffmpeg, imagemagick
    Databases (Clients): sqlite3, postgresql-client, default-mysql-client
    Dev Libraries: libssl-dev, zlib1g-dev, libffi-dev, uuid-dev, and various other headers (readline, sqlite3, etc.).
    2. Windows Compatibility (Wine)
    Wine: wine (64-bit), wine32 (32-bit architecture enabled), fonts-wine
    Utilities: winetricks, cabextract, zenity
    3. Python (Data Science & Web)
    Core: Python 3 (full), pip, venv
    Data Science: numpy, pandas, scipy, matplotlib, seaborn, scikit-learn
    Web Frameworks: flask, fastapi, uvicorn, django, requests
    Database/Cloud: boto3 (AWS), sqlalchemy, psycopg2-binary
    Tools: pytest, black, flake8, ipython, jupyterlab, beautifulsoup4, lxml, pyyaml, pillow, openpyxl
    4. JavaScript / TypeScript
    Runtimes: Node.js 22 (LTS), Bun
    Package Managers: npm (latest), yarn, pnpm
    Global Tools: typescript, ts-node, nodemon, eslint, prettier
    Framework CLIs: @angular/cli, react-scripts, express-generator
    5. Rust
    Language: Rust (installed via rustup)
    Cargo Tools: ripgrep (fast grep), bat (cat clone), fd-find (find clone)
    6. Go (Golang)
    Language: Go version 1.23.4
    7. Java
    JDK: OpenJDK 21
    Build Tools: maven, gradle
    8. .NET
    SDK: .NET 8.0 SDK
    
    You may use any of the tooling and install your own if needed. 
    Before using any tool, even if it's on the list, you must check if it's installed and if not the script must be able to handle it and install it.

    Goal:
    Create a production-grade automated installation script and a reviewer guide.

    Task 1: bash install script
    Write a Bash script to install and run the project. You must adhere to the following strict coding standards:
    1.  **Strict Mode & Safety:** Start with `set -euo pipefail` to ensure the script fails instantly on errors or undefined variables.
    2.  **Visual Logging:** Use the following function style for output (Green for INFO, Yellow for WARNING, Red for ERROR):
        - `print_status() {{ echo -e "\\033[0;32m[INFO]\\033[0m $1"; }}`
        - `print_error() {{ echo -e "\\033[0;31m[ERROR]\\033[0m $1"; }}`
    3.  **Error Handling:** Use a `trap` function to catch errors and print a helpful message before exiting.
    4.  **Apt Reliability:** Before running `apt-get install`, use a loop to check for and wait on `/var/lib/dpkg/lock` to ensure apt is not locked by background processes.
    5.  **Idempotency:** Do not blind install. Check if a package/tool exists using `command -v` before attempting to install it.
    6.  **Environment Isolation:** If Python is required, create a virtual environment (`python3 -m venv venv`), activate it, and install requirements there. Do NOT install global pip packages.
    7.  **Execution:** The script must handle all dependencies and end by running the project (or printing the command to run it if it is a service).

    Task 2: Markdown reviewer guide
    Create a Markdown guide.
    1.  **Header:** The exact command to execute the project (e.g., `./airlock_install.sh` or `source venv/bin/activate && python main.py`).
    2.  **Summary:** A concise technical summary of the project's purpose and the tech stack found in the file structure.
    3.  **Installer Logic:** A technical explanation of what the script does (e.g., "Checks apt locks, ensures Python 3.10+, creates a virtual environment, installs requirements.txt...").

    Task 3: Tech Stack
    A 1-line comma-separated list of the specific languages, frameworks, and critical tools detected.

    Task 4: Summary
    A 2-line summary. Line 1: What the repo does. Line 2: How the install script achieves the setup.

    Output Format:
    Return ONLY valid JSON with no markdown formatting.
    IMPORTANT: Ensure all strings are properly escaped (especially newlines and quotes) to be valid JSON.
    {{
      "script": "code string",
      "guide": "markdown string",
      "tech_stack": "plaintext string",
      "summary": "plaintext string"
    }}
    """
    
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            payload = {
                "model": model_id,
                "messages": [{"role": "system", "content": prompt}],
                "temperature": 0.5,
                "stream": True
            }

            full_content = ""
            start_time = asyncio.get_event_loop().time()
            last_update_time = start_time
            token_count = 0
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", "https://ai.hackclub.com/proxy/v1/chat/completions",
                                      headers={"Authorization": f"Bearer {HC_AI_KEY}"}, json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                            
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk['choices'][0]['delta'].get('content', '')
                            full_content += delta
                            token_count += 1 # Rough estimation
                            
                            current_time = asyncio.get_event_loop().time()
                            if current_time - last_update_time >= 2:
                                elapsed = current_time - start_time
                                tps = token_count / elapsed if elapsed > 0 else 0
                                yield f"    [{int(elapsed)}s] Status: {token_count} tokens {tps:.1f} tok/s\n"
                                last_update_time = current_time
                                
                        except:
                            continue

            # Process the full content
            content = full_content
            
            # Clean up markdown code blocks if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # JSON parsing failed, try to extract fields using regex as a fallback
                import re
                
                script_match = re.search(r'"script"\s*:\s*"(.*?)(?<!\\)"', content, re.DOTALL)
                guide_match = re.search(r'"guide"\s*:\s*"(.*?)(?<!\\)"', content, re.DOTALL)
                tech_stack_match = re.search(r'"tech_stack"\s*:\s*"(.*?)(?<!\\)"', content, re.DOTALL)
                summary_match = re.search(r'"summary"\s*:\s*"(.*?)(?<!\\)"', content, re.DOTALL)
                
                if script_match and guide_match and tech_stack_match and summary_match:
                     # Unescape the extracted strings
                     def unescape_json_string(s):
                         return s.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')

                     data = {
                         'script': unescape_json_string(script_match.group(1)),
                         'guide': unescape_json_string(guide_match.group(1)),
                         'tech_stack': unescape_json_string(tech_stack_match.group(1)),
                         'summary': unescape_json_string(summary_match.group(1))
                     }
                else:
                    logger.warning(f"AI Response content (failed to parse): {content[:500]}...")
                    raise

            script = data.get('script', '')
            guide = data.get('guide', '')
            tech_stack = data.get('tech_stack', '')
            summary = data.get('summary', '')

            html = f"<html><title>Airlock Manual</title><body style='font-family:sans-serif;padding:20px'><h1>Airlock Manual</h1><h2>AI Review Guide</h2>{markdown.markdown(guide)}<hr><h2>Vibecoded Install script</h2><pre><code>{script}</code></pre><p>You can run it with <code>bash ./airlock_install.sh</code></p><hr><h2>Airlock Info</h2><p>Airlock is a Hack Club tool for reviewing code in an ephemeral virtualized environment. Airlock sessions may not last longer than 1 hour. Please remember to close the Airlock session once you are done. You can use Airlock on airlock.hackclub.com. If you experience any issues, please contact @Carlos on Slack.</p></body></html>"

            yield {"type": "result", "data": (script, html, tech_stack, summary)}
            return
        
        except Exception as e:
            last_exception = e
            logger.warning(f"AI Attempt {attempt+1}/{max_retries+1} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(1) # Short backoff
                continue

    raise last_exception

async def create_kasm_session_generator(repo_url, repo_name, install_script, help_html, tech_stack, kasm_user_id=None):
    yield "[*] 1. Requesting Session...\n"

    payload = {
        "api_key": KASM_API_KEY,
        "api_key_secret": KASM_API_SECRET,
        "image_id": KASM_IMAGE_ID,
        "enable_sharing": True,
        "run_config": {"skip_check": True},
        "launch_config": {
            "git_url": repo_url
        }
    }

    if kasm_user_id:
        payload["user_id"] = kasm_user_id

    async with httpx.AsyncClient(verify=False) as client:
        resp = await client.post(f"{KASM_URL}/api/public/request_kasm", json=payload)
        data = resp.json()

        kasm_id = data.get('kasm_id')
        user_id = data.get('user_id')

        if not kasm_id or not user_id:
            yield f"\n[!] FATAL ERROR: Airlock refused to create session. Please contact @Carlos on Slack\n"
            yield f"    Server Response: {json.dumps(data, indent=2)}\n"
            return

        yield f"    > Session ID: {kasm_id}\n"
        yield f"    > User ID: {user_id}\n"

        yield "[*] 2. Waiting for container to start...\n"
        ready = False
        kasm_url = ""

        for i in range(150):
            await asyncio.sleep(2)
            try:
                status_resp = await client.post(f"{KASM_URL}/api/public/get_kasm_status",
                                          json={"api_key": KASM_API_KEY, "api_key_secret": KASM_API_SECRET, "kasm_id": kasm_id, "user_id": user_id})

                if status_resp.status_code == 200:
                    s_data = status_resp.json()

                    op_status = s_data.get('operational_status')
                    op_msg = s_data.get('operational_message', '')

                    if 'kasm' in s_data and s_data['kasm']:
                        nested_status = s_data['kasm'].get('operational_status')
                        if nested_status:
                            op_status = nested_status

                        if not s_data.get('kasm_url'):
                            kasm_url = s_data['kasm'].get('kasm_url', '')
                        else:
                            kasm_url = s_data.get('kasm_url')

                    if not op_msg: op_msg = "Processing..."

                    yield f"    [{i*2}s] Status: {op_status} | Msg: {op_msg}\n"

                    if op_status == 'running':
                        if not kasm_url and s_data.get('kasm_url'):
                            kasm_url = s_data.get('kasm_url')
                        ready = True
                        break

                    if op_status in ['stopped', 'failed']:
                        yield f"Session failed to start. Status: {op_status}\n"
                        return
            except Exception as e:
                yield f"Error polling status: {e}\n"

        if not ready:
            yield "Container timed out starting.\n"
            return

        yield "[*] 3. Injecting AI Scripts & Apps...\n"
        b64_script = base64.b64encode(install_script.encode('utf-8')).decode('utf-8')
        b64_html = base64.b64encode(help_html.encode('utf-8')).decode('utf-8')

        cmd = (
            f"bash -c '"
            f"pkill zenity; "
            f"echo \"kasm-user ALL=(ALL) NOPASSWD: ALL\" > /etc/sudoers.d/kasm-nopasswd && chmod 0440 /etc/sudoers.d/kasm-nopasswd; "
            
            # Set global environment variables
            f"echo \"export BUN_INSTALL=/usr/local\" >> /etc/profile.d/airlock_env.sh; "
            f"echo \"export RUSTUP_HOME=/opt/rust\" >> /etc/profile.d/airlock_env.sh; "
            f"echo \"export CARGO_HOME=/opt/rust\" >> /etc/profile.d/airlock_env.sh; "
            f"echo \"export PATH=\\\"/opt/rust/bin:/usr/local/go/bin:$PATH\\\"\" >> /etc/profile.d/airlock_env.sh; "
            f"chmod +x /etc/profile.d/airlock_env.sh; "

            # Also export them for the current root session so subsequent commands use them if needed
            f"export BUN_INSTALL=/usr/local; "
            f"export RUSTUP_HOME=/opt/rust; "
            f"export CARGO_HOME=/opt/rust; "
            f"export PATH=\"/opt/rust/bin:/usr/local/go/bin:$PATH\"; "

            f"export DISPLAY=:1; "
            f"sudo -u kasm-user git clone \"{repo_url}\" \"/home/kasm-user/Desktop/{repo_name}\"; "
            f"echo \"{b64_script}\" | base64 -d > \"/home/kasm-user/Desktop/{repo_name}/airlock_install.sh\"; "
            f"echo \"{b64_html}\" | base64 -d > \"/home/kasm-user/Desktop/{repo_name}/REVIEW_GUIDE.html\"; "
            f"chmod +x \"/home/kasm-user/Desktop/{repo_name}/airlock_install.sh\"; "
            f"sudo -u kasm-user DISPLAY=:1 x-www-browser \"file:///home/kasm-user/Desktop/{repo_name}/REVIEW_GUIDE.html\" >/dev/null 2>&1 & "
            f"sudo -u kasm-user DISPLAY=:1 x-www-browser \"{repo_url}\" >/dev/null 2>&1 & "
            f"sudo -u kasm-user DISPLAY=:1 thunar \"/home/kasm-user/Desktop/{repo_name}\" & "
            f"sudo -u kasm-user DISPLAY=:1 wget https://hc-cdn.hel1.your-objectstorage.com/s/v3/71cb3c1895619d1990e54f5b3ac32e79b80be018_airlock_background.png -O /usr/share/backgrounds/bg_default.png & "
            f"sudo -u kasm-user DISPLAY=:1 xfce4-terminal --working-directory=\"/home/kasm-user/Desktop/{repo_name}\" -x bash -c \"source /etc/profile.d/airlock_env.sh; ls -lah; exec bash\" & "
            f"sudo -u kasm-user DISPLAY=:1 xfce4-terminal --working-directory=\"/home/kasm-user/Desktop/{repo_name}\" -x bash -c \"source /etc/profile.d/airlock_env.sh; bash ./airlock_install.sh; exec bash\" & "
            f"'"
        )

        exec_payload = {
            "api_key": KASM_API_KEY,
            "api_key_secret": KASM_API_SECRET,
            "kasm_id": kasm_id,
            "user_id": user_id,
            "exec_config": {
                "cmd": cmd,
                "user": "root"
            }
        }

        await client.post(f"{KASM_URL}/api/public/exec_command_kasm", json=exec_payload)

        if kasm_url and kasm_url.startswith("/"):
            kasm_url = f"{KASM_URL}{kasm_url}"

        yield f"[SUCCESS] Session: {kasm_url}\n"

# --- Dependencies ---

async def get_current_user(request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user

async def get_admin_user(user: dict = Depends(get_current_user)):
    if not is_admin(user['slack_id']):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

async def get_org_admin_user(user: dict = Depends(get_current_user)):
    if is_admin(user['slack_id']) or is_any_org_admin(user['slack_id']):
        return user
    raise HTTPException(status_code=403, detail="Organization Admin access required")

# --- Routes ---

@app.get("/")
async def root():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/admin")
async def admin_page():
    with open("static/admin.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for('auth_callback')
    return await oauth.hackclub.authorize_redirect(request, redirect_uri)

@app.get("/auth")
async def auth_callback(request: Request):
    try:
        token = await oauth.hackclub.authorize_access_token(request)
        user_info = token.get('userinfo')
        if not user_info:
             user_info = await oauth.hackclub.userinfo(token=token)

        # Fallback: If slack_id missing, try /api/v1/me
        if not user_info.get('slack_id'):
            access_token = token.get('access_token')
            async with httpx.AsyncClient() as client:
                r = await client.get("https://auth.hackclub.com/api/v1/me",
                                     headers={"Authorization": f"Bearer {access_token}"})
                if r.status_code == 200:
                    api_user = r.json()
                    if 'identity' in api_user and 'slack_id' in api_user['identity']:
                        user_info['slack_id'] = api_user['identity']['slack_id']
                        if 'name' not in user_info and 'name' in api_user['identity']:
                             user_info['name'] = api_user['identity'].get('name', 'User')
                        if 'email' not in user_info and 'primary_email' in api_user['identity']:
                             user_info['email'] = api_user['identity']['primary_email']

        slack_id = user_info.get('slack_id')
        if not slack_id:
             return HTMLResponse(
                 """
                 <html>
                 <head>
                    <title>Airlock | Access Denied</title>
                    <style>
                        body { display: flex; justify-content: center; align-items: center; min-height: 100vh; flex-direction: column; margin: 0; font-family: sans-serif; padding: 40px; text-align: center; }
                        .btn { display: inline-block; background: #ec3750; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; margin-top: 20px; }
                    </style>
                </head>
                <body>
                    <h1>Access Denied</h1>
                    <p style="margin: 0px;">We could not find a Slack ID associated with your profile.</p>
                    <p>Please connect your Slack account to your Hack Club account.</p>
                    <a href="https://auth.hackclub.com/" class="btn">Connect Slack on Hack Club Auth</a>
                </body>
                </html>
                 """,
                 status_code=403
             )

        if not is_authorized(slack_id):
             return HTMLResponse(
                 f"""
                 <title>Airlock | Access Denied</title><style>@font-face{{font-family:'Phantom Sans';src:url(https://assets.hackclub.com/fonts/Phantom_Sans_0.7/Bold.woff) format('woff'),url(https://assets.hackclub.com/fonts/Phantom_Sans_0.7/Bold.woff2) format('woff2');font-weight:700;font-style:normal;font-display:swap}}body{{font-family:sans-serif;margin:0;padding:0;display:flex;justify-content:center;align-items:center;min-height:100vh;background-color:#f4f4f4;color:#333}}.container{{background:#fff;padding:2.5rem;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,.1);max-width:700px;margin:40px 20px}}h1{{font-family:'Phantom Sans',sans-serif;color:#ec3750;margin-top:0;margin-bottom:1rem}}p{{line-height:1.6;margin:1rem 0}}ul{{text-align:left;line-height:1.8;margin:1.5rem 0;padding-left:1.5rem}}li{{margin-bottom:1rem;padding-left:.5rem}}strong{{color:#ec3750;font-weight:600}}a{{display:inline-block;background-color:#ec3750;color:#fff;text-decoration:none;padding:12px 24px;border-radius:4px;margin-top:1.5rem;font-family:'Phantom Sans',sans-serif;transition:background-color .2s}}a:hover{{background-color:#d12e43}}</style><div class=container><h1>Access Denied</h1><p>User <strong>{slack_id}</strong> is not authorized to use Airlock.<p>There are a few possible reasons for this:<ul><li>You shouldn't be here, but you saw Hack Club page with log in screen and you clicked the button.<li>You are a reviewer and have a reason to use Airlock, but either you haven't been added to your organization or your organization has not been created yet. If this is the case, please contact the admin of your Airlock Organization.<li>You are the lead reviewer/hq contact/something else on your YSWS/event. But you don't have an organization set up on Airlock yet. If this is the case, please contact @Carlos on Slack or fill out the form below.<li>You do have access to Airlock, but you are logged in with the wrong Slack ID or the wrong Slack ID has been added to your organization. If this is the case, please confirm <strong>{slack_id}</strong> is the correct Slack ID.</ul><p><a href="https://forms.hackclub.com/airlock?id={slack_id}">Click here to request an organization on Airlock</a></div>
                 """,
                 status_code=403
             )

        # Add is_admin flag to session
        user_info['is_admin'] = is_admin(slack_id)
        user_info['is_org_admin'] = is_any_org_admin(slack_id)
        request.session['user'] = dict(user_info)

        return RedirectResponse(url='/')
    except Exception as e:
        logger.exception("Auth failed")
        return HTMLResponse(f"<h1>Auth Failed: {e}</h1>", status_code=400)

@app.get("/logout")
async def logout(request: Request):
    request.session.pop('user', None)
    return RedirectResponse(url='/')

@app.get("/api/v1/me")
async def me(user: dict = Depends(get_current_user)):
    user_data = dict(user)
    org = get_user_organization(user['slack_id'])
    if org:
        user_data['organization'] = org
    return {"status": "authenticated", "user": user_data}

@app.get("/api/v1/getSession")
async def get_session(repo_url: str, user: dict = Depends(get_current_user)):
    # Determine KASM user ID
    kasm_user_id = None
    org = get_user_organization(user['slack_id'])
    if org:
        kasm_user_id = org.get('kasm_user_id')
    
    async def process_stream():
        try:
            # URL Validation & Normalization
            try:
                normalized_url, warning = normalize_github_url(repo_url)
                if warning:
                    yield f"[!] {warning}\n"
                repo_url_final = normalized_url
            except ValueError as ve:
                yield f"[!] Error: {str(ve)}\n"
                return

            yield f"[*] Fetching GitHub data for {repo_url_final}...\n"
            try:
                context = await get_github_context_data(repo_url_final)
            except Exception as e:
                yield f"[!] Error fetching GitHub data: {e}\n"
                return

            # Pre-analysis
            yield "[*] Inspecting the project... (google/gemini-2.5-flash-lite)\n"
            try:
                pre_analysis = await pre_analyze_project(context)
                tech_stack = pre_analysis.get('tech_stack', 'Unknown')
                difficulty = pre_analysis.get('difficulty', 'easy')
                files_to_read = pre_analysis.get('files_to_read', [])
                
                yield f"\n[*] Detected Tech Stack: {tech_stack}\n"
            except Exception as e:
                yield f"[!] Pre-analysis failed: {e}\n"
                # Fallback defaults
                difficulty = 'easy'
                files_to_read = []
                tech_stack = 'Unknown'

            # Fetch extra files
            additional_content = ""
            if files_to_read:
                yield f"[*] Reading: {', '.join(files_to_read)}...\n"
                file_contents = []
                max_total_chars = 4000
                max_per_file = max_total_chars // len(files_to_read) if files_to_read else 4000
                
                for fpath in files_to_read:
                    content = await fetch_github_file_content(context['owner'], context['repo'], fpath)
                    if content:
                        truncated = content[:max_per_file]
                        file_contents.append(f"File: {fpath}\nContent:\n{truncated}\n")
                
                additional_content = "\n".join(file_contents)

            # Determine model
            model_id = "z-ai/glm-4.6"
            if difficulty == "hard":
                model_id = "google/gemini-3-pro-preview"
            
            yield f"[*] Analyzing with AI... ({model_id})\n"
            
            try:
                script = ""
                html = ""
                final_tech_stack = ""
                summary = ""
                
                async for chunk in analyze_with_ai_data(context, additional_content, model_id):
                    if isinstance(chunk, str):
                        yield chunk
                    elif isinstance(chunk, dict) and chunk['type'] == 'result':
                        script, html, final_tech_stack, summary = chunk['data']

                yield f"\n[*] AI Summary: {summary}\n"
            except Exception as e:
                yield f"[!] AI Analysis failed: {e}\n"
                return

            async for msg in create_kasm_session_generator(repo_url_final, context['name'], script, html, tech_stack, kasm_user_id=kasm_user_id):
                yield msg

        except Exception as e:
            yield f"\n[ERROR] {e}\n"

    return StreamingResponse(process_stream(), media_type="text/plain")

# --- Admin API ---

@app.get("/api/v1/admin/users")
async def list_users(admin: dict = Depends(get_admin_user)):
    return load_users()

@app.post("/api/v1/admin/users")
async def add_user(user: User, admin: dict = Depends(get_admin_user)):
    users = load_users()
    if any(u['slack_id'] == user.slack_id for u in users):
        raise HTTPException(status_code=400, detail="User already exists")

    users.append(user.dict())
    save_users(users)
    return {"status": "added", "user": user}

@app.delete("/api/v1/admin/users/{slack_id}")
async def delete_user(slack_id: str, admin: dict = Depends(get_admin_user)):
    users = load_users()
    users = [u for u in users if u['slack_id'] != slack_id]
    save_users(users)
    return {"status": "deleted"}

# --- Organization API ---

@app.get("/api/v1/admin/organizations")
async def list_organizations(admin: dict = Depends(get_admin_user)):
    return load_organizations()

@app.post("/api/v1/admin/organizations")
async def create_organization(org: Organization, admin: dict = Depends(get_admin_user)):
    orgs = load_organizations()
    if any(o['id'] == org.id for o in orgs):
        raise HTTPException(status_code=400, detail="Organization ID already exists")
    
    # Generate ID if not provided (though model requires it currently, better to handle it)
    if not org.id:
        org.id = str(uuid.uuid4())

    orgs.append(org.dict())
    save_organizations(orgs)
    return {"status": "created", "organization": org}

@app.get("/api/v1/admin/organizations/{org_id}")
async def get_organization(org_id: str, user: dict = Depends(get_org_admin_user)):
    # Check permission
    if not is_admin(user['slack_id']) and not is_org_admin(user['slack_id'], org_id):
        raise HTTPException(status_code=403, detail="Access denied to this organization")

    orgs = load_organizations()
    org = next((o for o in orgs if o['id'] == org_id), None)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org

@app.put("/api/v1/admin/organizations/{org_id}")
async def update_organization(org_id: str, update: OrganizationUpdate, admin: dict = Depends(get_admin_user)):
    orgs = load_organizations()
    for i, o in enumerate(orgs):
        if o['id'] == org_id:
            if update.name is not None:
                o['name'] = update.name
            if update.kasm_user_id is not None:
                o['kasm_user_id'] = update.kasm_user_id
            if update.session_limit is not None:
                o['session_limit'] = update.session_limit
            
            orgs[i] = o
            save_organizations(orgs)
            return {"status": "updated", "organization": o}
            
    raise HTTPException(status_code=404, detail="Organization not found")

@app.post("/api/v1/admin/organizations/{org_id}/users")
async def add_org_user(org_id: str, data: dict, user: dict = Depends(get_org_admin_user)):
    # data: { "slack_id": "...", "role": "admin" | "user" }
    slack_id = data.get("slack_id")
    role = data.get("role", "user")
    
    if not slack_id:
        raise HTTPException(status_code=400, detail="slack_id is required")

    # Check permission
    if not is_admin(user['slack_id']) and not is_org_admin(user['slack_id'], org_id):
        raise HTTPException(status_code=403, detail="Access denied to this organization")

    orgs = load_organizations()
    org_idx = next((i for i, o in enumerate(orgs) if o['id'] == org_id), -1)
    
    if org_idx == -1:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    org = orgs[org_idx]
    
    # Remove from both lists first to ensure no duplicates/conflicts
    if slack_id in org['admins']:
        org['admins'].remove(slack_id)
    if slack_id in org['users']:
        org['users'].remove(slack_id)
        
    if role == 'admin':
        org['admins'].append(slack_id)
    else:
        org['users'].append(slack_id)
        
    orgs[org_idx] = org
    save_organizations(orgs)
    
    # Fetch profile for return
    profile = get_slack_profile(slack_id)
    return {"status": "added", "role": role, "profile": profile}

@app.delete("/api/v1/admin/organizations/{org_id}/users/{slack_id}")
async def remove_org_user(org_id: str, slack_id: str, user: dict = Depends(get_org_admin_user)):
    # Check permission
    if not is_admin(user['slack_id']) and not is_org_admin(user['slack_id'], org_id):
        raise HTTPException(status_code=403, detail="Access denied to this organization")

    orgs = load_organizations()
    org_idx = next((i for i, o in enumerate(orgs) if o['id'] == org_id), -1)
    
    if org_idx == -1:
        raise HTTPException(status_code=404, detail="Organization not found")

    org = orgs[org_idx]
    
    if slack_id in org['admins']:
        org['admins'].remove(slack_id)
    if slack_id in org['users']:
        org['users'].remove(slack_id)
        
    orgs[org_idx] = org
    save_organizations(orgs)
    return {"status": "removed"}

@app.get("/api/v1/slack/profile/{slack_id}")
async def get_profile(slack_id: str, response: Response, user: dict = Depends(get_current_user)):
    response.headers["Cache-Control"] = "private, max-age=86400"
    return get_slack_profile(slack_id)
