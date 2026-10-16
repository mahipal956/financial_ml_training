"""
Gemini GitHub Automation Assistant
Reads user request, asks Gemini model for code changes, applies them, and prepares a branch.
"""
import os
import sys
import json
import subprocess
import requests

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}\n{result.stderr}")
    return result.stdout.strip()

def get_gemini_response(prompt, context):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is missing.")
        sys.exit(1)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent?key={api_key}"
    
    system_instruction = (
        "You are an expert autonomous software engineer. "
        "The user will give you a task and repository context. "
        "Return your response ONLY in valid JSON format matching this schema: \n"
        "{\n"
        '  "summary": "Brief description of changes made",\n'
        '  "files": [\n'
        '    {"path": "relative/path/to/file.ext", "content": "full new content of the file"}\n'
        "  ]\n"
        "}\n"
        "Do not include any markdown fences or extra prose outside the JSON."
    )

    full_prompt = f"TASK:\n{prompt}\n\nREPOSITORY CONTEXT:\n{context}"

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2
        }
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        print(f"Gemini API Error ({response.status_code}): {response.text}")
        sys.exit(1)

    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)

def main():
    task_prompt = os.environ.get("TASK_PROMPT", "Optimize code and add unit tests")
    print(f"Processing task: {task_prompt}")

    # Gather list of files in repo
    file_list = run_cmd("git ls-files").splitlines()
    code_files = [f for f in file_list if f.endswith(('.py', '.md', '.txt', '.yaml', '.json')) and not f.startswith('.github/')]
    
    # Read a sample of main project files for context
    context = ""
    for f in code_files[:15]:
        if os.path.exists(f) and os.path.getsize(f) < 50000:
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                    context += f"\n--- FILE: {f} ---\n" + fp.read()[:3000] + "\n"
            except Exception as e:
                pass

    print("Querying Gemini API for code solution...")
    result = get_gemini_response(task_prompt, context)

    summary = result.get("summary", "Automated code updates from Gemini")
    files_to_update = result.get("files", [])

    print(f"Gemini Summary: {summary}")
    print(f"Modifying {len(files_to_update)} files...")

    for item in files_to_update:
        file_path = item["path"]
        content = item["content"]
        
        # Security check: avoid directory traversal
        if file_path.startswith("/") or ".." in file_path:
            continue
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True) if os.path.dirname(file_path) else None
        with open(file_path, "w", encoding="utf-8") as fp:
            fp.write(content)
        print(f"Updated: {file_path}")

    # Save summary for GitHub Actions PR output
    with open("gemini_pr_summary.txt", "w", encoding="utf-8") as fp:
        fp.write(summary)

if __name__ == "__main__":
    main()
