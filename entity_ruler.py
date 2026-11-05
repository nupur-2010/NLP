"""
EntityRuler - Custom EntityRuler patterns for ScamShield
Creates spaCy EntityRuler patterns for scam-related entities
"""

import spacy
import re


def create_phone_patterns():
    """Create patterns for various phone number formats"""
    patterns = [
        (r"\+91[\s\-]?\d{10}", "PHONE"),
        (r"\b\d{10}\b", "PHONE"),
        (r"\+\d{1,3}[\s\-]?\d{6,12}", "PHONE"),
        (r"\b\d{3}[\-\s]?\d{3}[\-\s]?\d{4}\b", "PHONE"),
        (r"\b\d{4}[\-\s]?\d{3}[\-\s]?\d{3}\b", "PHONE"),
        (r"\b1800[\-\s]?\d{3}[\-\s]?\d{3}\b", "PHONE"),
    ]
    return patterns


def create_email_patterns():
    """Create patterns for email addresses"""
    patterns = [
        (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "EMAIL"),
    ]
    return patterns


def create_url_patterns():
    """Create patterns for URLs"""
    patterns = [
        (r"https?://[^\s]+", "URL"),
        (r"www\.[^\s]+", "URL"),
    ]
    return patterns


def create_upi_patterns():
    """Create patterns for UPI IDs"""
    patterns = [
        (r"[a-zA-Z0-9._-]+@[a-zA-Z]+", "UPI"),
    ]
    return patterns


def create_currency_patterns():
    """Create patterns for currency amounts"""
    patterns = [
        (r"₹\s?\d+[\d,]*(\.\d{2})?", "CURRENCY"),
        (r"Rs\.?\s?\d+[\d,]*(\.\d{2})?", "CURRENCY"),
        (r"\$\s?\d+[\d,]*(\.\d{2})?", "CURRENCY"),
        (r"USD\s?\d+[\d,]*(\.\d{2})?", "CURRENCY"),
        (r"€\s?\d+[\d,]*(\.\d{2})?", "CURRENCY"),
        (r"\d+[\d,]*\s?(INR|USD|EUR|GBP)R?", "CURRENCY"),
    ]
    return patterns


def get_all_regex_patterns():
    """Combine all patterns"""
    return (
        create_phone_patterns() +
        create_email_patterns() +
        create_url_patterns() +
        create_upi_patterns() +
        create_currency_patterns()
    )


def setup_nlp_with_ruler():
    """
    Set up spaCy NLP pipeline (EntityRuler not used - using regex instead)
    
    Returns:
        Configured spacy nlp object
    """
    nlp = spacy.load("en_core_web_sm")
    return nlp


def extract_custom_entities(text):
    """Extract custom entities using regex patterns"""
    entities = []
    for pattern, label in get_all_regex_patterns():
        for match in re.finditer(pattern, text):
            entities.append({
                "text": match.group(),
                "label": label,
                "start": match.start(),
                "end": match.end()
            })
    # Sort by start position
    entities.sort(key=lambda x: x["start"])
    return entities


if __name__ == "__main__":
    nlp = setup_nlp_with_ruler()
    
    test_texts = [
        "Call me at +91 9876543210 or email rahul@example.com",
        "Send ₹5000 to name@upi",
        "Visit https://fake-bank.com/verify",
        "You won $1,000,000! Claim at www.prize.com",
    ]
    
    print("Testing regex patterns:\n")
    for text in test_texts:
        print(f"Text: {text}")
        entities = extract_custom_entities(text)
        print(f"Custom Entities: {[(e['text'], e['label']) for e in entities]}")
        # Also show spaCy entities
        doc = nlp(text)
        print(f"Spacy Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")
        print()