# app/settings.py
import os

# Try to load .env if python-dotenv is installed; otherwise it's a no-op.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

class Settings:
    def __init__(self) -> None:
        # Reads from environment or .env (if loaded)
        self.GOOGLE_MAPS_API_KEY: str = os.getenv("GOOGLE_MAPS_API_KEY", "")

settings = Settings()
