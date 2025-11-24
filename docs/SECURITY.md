# Security Documentation

This document describes the security measures implemented in the nba_api package.

## Security Fixes (v1.11.3+)

### 1. File I/O Vulnerabilities Fixed

**Location:** `src/nba_api/library/http.py`

**Issue:** Original code used raw `open()` calls without context managers, risking resource leaks.

**Fix:**
- All file operations now use context managers (`with` statements)
- Debug storage moved from source tree to system temp directory
- Restrictive permissions (0o700) on debug storage directory
- Security warning when DEBUG_STORAGE is enabled

```python
# Before (vulnerable)
f = open(file_path, "r")
contents = f.read()
f.close()

# After (secure)
with open(file_path, "r", encoding="utf-8") as f:
    contents = f.read()
```

### 2. Endpoint Injection Prevention

**Location:** `src/nba_api/library/http.py`

**Issue:** Endpoint names were not validated, potentially allowing injection attacks.

**Fix:** Added regex validation for endpoint names.

```python
VALID_ENDPOINT_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')

# Validates endpoint before use
if not VALID_ENDPOINT_PATTERN.match(endpoint):
    raise ValueError(f"Invalid endpoint name: {endpoint}")
```

### 3. ReDoS (Regular Expression Denial of Service) Protection

**Location:** `src/nba_api/stats/static/players.py`

**Issue:** User-provided regex patterns in player search could cause ReDoS attacks.

**Fix:**
- Added `safe_search` parameter to escape user input
- Added pattern length limits (200 characters max)
- Added regex error handling

```python
# Safe search (escapes special characters)
players = find_players_by_full_name("O'Neal", safe_search=True)

# Pattern search (use with trusted input only)
players = find_players_by_full_name("Lebron|Kobe", safe_search=False)
```

### 4. Debug Storage Security

**Location:** `src/nba_api/library/http.py`

**Issue:** Debug storage was in the source tree and could leak sensitive data.

**Fix:**
- Moved to system temp directory (`/tmp/nba_api_debug_storage/`)
- Added security warning when enabled
- Restrictive file permissions

```python
# Storage location (secure temp directory)
file_path = os.path.join(tempfile.gettempdir(), "nba_api_debug_storage")
os.makedirs(file_path, mode=0o700)  # Owner access only
```

## Security Best Practices

### For Users

1. **Never enable DEBUG_STORAGE in production**
   ```python
   # In nba_api/library/debug/debug.py
   DEBUG = False
   DEBUG_STORAGE = False  # Keep this False in production
   ```

2. **Use safe_search for user input**
   ```python
   # When search pattern comes from user input
   players = find_players_by_full_name(user_input, safe_search=True)
   ```

3. **Validate player/team IDs before use**
   ```python
   player_id = predictor.get_player_id(user_input)
   if player_id is None:
       raise ValueError("Invalid player name")
   ```

### For Developers

1. **Use context managers for file operations**
   ```python
   with open(filepath, 'r', encoding='utf-8') as f:
       data = f.read()
   ```

2. **Validate all external input**
   - Endpoint names
   - Search patterns
   - File paths

3. **Use parameterized queries for database operations**
   ```python
   # Correct
   cursor.execute("SELECT * FROM players WHERE id = ?", (player_id,))

   # Wrong (SQL injection risk)
   cursor.execute(f"SELECT * FROM players WHERE id = {player_id}")
   ```

## Reporting Security Issues

If you discover a security vulnerability, please report it via:
- GitHub Security Advisories: https://github.com/swar/nba_api/security/advisories
- Email: security@example.com

Please do not create public issues for security vulnerabilities.

## Dependency Security

The package uses these security-related practices:

1. **Snyk scanning** in CI/CD pipeline
2. **Dependabot** for dependency updates
3. **Pinned versions** in pyproject.toml
4. **Regular security audits** of dependencies

### Current Dependencies

| Package | Version | Security Notes |
|---------|---------|----------------|
| requests | >=2.32.3 | Recent version with security patches |
| numpy | >=1.26.0 | Updated for Python 3.10+ |
| pandas | >=2.1.0 | Modern version |

## SSL/TLS Security

All API requests use HTTPS:
- `https://stats.nba.com/stats/`
- `https://cdn.nba.com/static/json/liveData/`

The package uses the default SSL certificate verification from the `requests` library.
