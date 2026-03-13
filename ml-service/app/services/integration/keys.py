import os
import pathlib


def _load_env_file():
    """Parse .env file and inject into os.environ."""
    candidates = [
        pathlib.Path(".env"),
        pathlib.Path(__file__).parent.parent / ".env",        
        pathlib.Path(__file__).parent.parent.parent / ".env", 
    ]
    for env_file in candidates:
        if env_file.exists():
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and val and key not in os.environ:
                        os.environ[key] = val
            break  # stop at first found .env


_load_env_file()

# Keys are read from environment (populated from .env above)
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

if not COHERE_API_KEY:
    print("⚠ WARNING: COHERE_API_KEY not found in .env")
if not GEMINI_API_KEY:
    print("⚠ WARNING: GEMINI_API_KEY not found in .env")