# STRIDE Threat Modeling Assessment: Roving Bard

This document presents a threat modeling assessment of the **Roving Bard** music player system based on the STRIDE framework.

---

## 1. System Boundaries and Data Flow Mapping

### Entry Points
- **REST API Routes (FastAPI)**: Serving routes like `/api/status`, `/api/control`, `/api/config`, `/api/screenshot`, etc.
- **Web GUI (`/gui`)**: Unprotected HTML interface loaded in the browser.
- **Agent API (`/run_sse`)**: Stream interface for AI pairing.
- **Mini-map Capture Pipeline**: Automated screenshot-taking utilizing `mss`, local image caching, OCR (`pytesseract`), and LiteLLM/Gemini Vision API fallback.

### Data Storage & External Boundaries
- **Playlist Directory (`music/`)**: Audio files stored locally on disk.
- **Capture Directory (`capture/`)**: Screenshot cached images.
- **Configuration (`mapping.yaml`)**: Stores active state minimap coordinates boundary mapping.
- **External APIs**: LiteLLM/Gemini endpoints using global environment API keys.

---

## 2. STRIDE Threat Assessment

### 👥 Spoofing
- **Vulnerability**: Authentication relies on verifying the `X-API-Key` header or `api_key` query parameter against environment variables (`AGENT_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_API_KEY`). If these variables are not configured in the host environment, any unauthenticated agent or user could bypass keys validation.
- **Access to Sensitive Tools**: Bypassing API key checks allows spoofing control commands, enabling full access to system commands (e.g. screenshot trigger, configuration alteration).

### ✏️ Tampering
- **Vulnerability (Path Traversal in Playback)**: The track selection in `play_track()` ([player.py](file:///home/chuubi/Desktop/vibe-coding-2026/capstone/roving-bard/app/player.py#L46)) does not sanitize the `track_file` filename input. If a malicious client passes path traversal sequences (e.g., `../../`), they can reference files outside the designated `music/` playlist folder.
- **Vulnerability (Configuration Tampering)**: The `/api/config` REST endpoint accepts raw bounds and coordinate configurations. If a spoofed command overwrites `mapping.yaml` boundaries, it can disrupt the parsing engine.

### 📝 Repudiation
- **Vulnerability**: While the codebase outputs diagnostic messages to standard output (e.g. `[Playback] Resuming music`), there is no structured audit logging for administrative actions such as configuration changes (`/api/config`), file uploads (`/api/upload-audio`), or manual play/stop triggers.

### 🔍 Information Disclosure
- **CRITICAL Vulnerability (API Key Exposure)**: The `/gui` route ([fast_api_app.py](file:///home/chuubi/Desktop/vibe-coding-2026/capstone/roving-bard/app/fast_api_app.py#L214-L224)) is **completely unprotected** and served without requiring authentication or API key checks. When requested, it reads the active environment keys (`AGENT_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_API_KEY`) and embeds them directly in the served HTML content via `content.replace("{{API_KEY_PLACEHOLDER}}", api_key)`. Any user accessing `/gui` can inspect the page source to leak raw credentials.
- **Vulnerability (Unhandled Stack Traces)**: Upload errors or OCR parsing issues print raw system traceback details directly back to the client or console logs, potentially leaking host path structures and dependencies.

### 💥 Denial of Service (DoS)
- **Vulnerability (Unbounded File Uploads)**: The `/api/upload-audio` route ([fast_api_app.py](file:///home/chuubi/Desktop/vibe-coding-2026/capstone/roving-bard/app/fast_api_app.py#L340-L362)) reads the entire file directly into memory using `file.file.read()` without any limit on the file size. This could trigger Out of Memory (OOM) crashes or exhaust disk space on the host.
- **Vulnerability (Uncapped API Consumption)**: The active screen scan functionality triggers Vision AI queries. If flooded by automated scripts, it can lead to rate-limit lockouts or large unexpected consumption of LLM credits.

### 👑 Elevation of Privilege
- **Vulnerability**: If key management falls back to unconfigured or default states, an unprivileged user on the local network can access administrative routes like `/api/upload-audio` or `/api/config` to upload executable scripts (e.g. as `.mp4` or other accepted extensions) or manipulate core application state.

---

## 3. Actionable Security Recommendations

1. **Protect GUI API Key Injection**:
   - Do not serve raw secret keys inside the GUI HTML text.
   - Require authentication on the `/gui` route itself, or manage auth dynamically in the browser session storage.
2. **Implement Input Sanitization**:
   - Apply `os.path.basename()` to `track_file` parameters inside `SafeMusicPlayer.play_track()` to prevent directory traversal attacks.
3. **Add File Upload Limits**:
   - Enforce a strict file-size limit (e.g., max 50MB) on `/api/upload-audio` and process the file upload in chunks rather than reading it entirely into memory.
4. **Establish Security Audit Logs**:
   - Replace standard `print` statements with structured python logging that logs actor information for critical state-modifying requests.
