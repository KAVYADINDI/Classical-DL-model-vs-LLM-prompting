# preprocess/text_cleaning.py
import re

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)           # remove urls
    text = re.sub(r"[^a-z0-9\s\.\,']", " ", text) # keep basic punctuation optionally
    text = re.sub(r"\s+", " ", text).strip()
    return text
