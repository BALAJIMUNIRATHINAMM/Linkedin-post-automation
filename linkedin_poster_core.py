"""
linkedin_poster_core.py - optional compatibility shim

Replace these functions with your own backend implementation if preferred.
"""

from typing import Optional

def generate_article(prompt: str, model: str = "gemini-pro-latest", gemini_api_key: Optional[str] = None) -> str:
    # Simple placeholder — replace with real Gemini call if you want a dedicated backend file.
    return f"# Prompt\n{prompt}\n\nGenerated article placeholder. Replace with a real generator."

def post_to_linkedin(article_text: str, org_id: Optional[str] = None, linkedin_api_key: Optional[str] = None) -> dict:
    # Placeholder that doesn't perform network calls
    return {"status": "mocked", "org_id": org_id, "length": len(article_text)}
