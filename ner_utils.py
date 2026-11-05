"""
NER Utilities - Named Entity Recognition for ScamShield
Handles spaCy model loading and entity extraction
"""

import spacy
from typing import List, Dict, Any

from entity_ruler import setup_nlp_with_ruler, extract_custom_entities


ENTITY_COLORS = {
    "PERSON": "#FF6B6B",
    "ORG": "#4ECDC4",
    "GPE": "#45B7D1",
    "DATE": "#96CEB4",
    "TIME": "#FFEAA7",
    "MONEY": "#DDA0DD",
    "PHONE": "#FF8C00",
    "EMAIL": "#9B59B6",
    "URL": "#3498DB",
    "UPI": "#E74C3C",
    "CURRENCY": "#F39C12",
}


def load_model() -> spacy.Language:
    """Load the spaCy model with custom EntityRuler"""
    return setup_nlp_with_ruler()


def extract_entities(text: str, nlp: spacy.Language = None) -> List[Dict[str, Any]]:
    """Extract entities from text using spaCy NER and custom regex patterns"""
    if nlp is None:
        nlp = load_model()
    
    doc = nlp(text)
    
    entities = []
    
    # Add spaCy NER entities
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "color": ENTITY_COLORS.get(ent.label_, "#808080"),
        })
    
    # Add custom regex-based entities
    for ent in extract_custom_entities(text):
        ent["color"] = ENTITY_COLORS.get(ent["label"], "#808080")
        entities.append(ent)
    
    # Remove duplicates (prefer spaCy entities if overlap)
    seen = set()
    unique_entities = []
    for ent in entities:
        key = (ent["text"], ent["label"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(ent)
    
    return unique_entities


def get_entity_counts(entities: List[Dict]) -> Dict[str, int]:
    """Count entities by type"""
    counts = {}
    for ent in entities:
        label = ent["label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def get_entity_types() -> List[str]:
    """Get list of all entity types used in ScamShield"""
    return [
        "PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY",
        "PHONE", "EMAIL", "URL", "UPI", "CURRENCY"
    ]


def highlight_entities_in_text(text: str, entities: List[Dict[str, Any]]) -> str:
    """Create HTML-highlighted version of text with entities marked"""
    if not entities:
        return text
    
    sorted_ents = sorted(entities, key=lambda x: x["start"], reverse=True)
    
    highlighted = text
    for ent in sorted_ents:
        color = ent["color"]
        replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; color: white;" title="{ent["label"]}">{ent["text"]}</span>'
        highlighted = highlighted[:ent["start"]] + replacement + highlighted[ent["end"]:]
    
    return highlighted


def clean_text_for_display(text: str) -> str:
    """Clean text for safe HTML display"""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


if __name__ == "__main__":
    nlp = load_model()
    
    test_messages = [
        "URGENT: Your ICICI account has been locked. Call 9876543210 immediately to unlock.",
        "Congratulations! You've won ₹5,00,000 in the lottery. Contact claim@prize.com or visit https://win.in",
        "Hello, I'm calling from Amazon customer service. Your order #12345 will be delivered tomorrow.",
        "Send ₹2000 to name@ybl immediately or your account will be closed. Call +91-9876543210.",
    ]
    
    print("Testing NER Extraction:\n")
    for msg in test_messages:
        print(f"Message: {msg}")
        entities = extract_entities(msg, nlp)
        print(f"Entities found: {len(entities)}")
        for e in entities:
            print(f"  - {e['text']} ({e['label']})")
        print()
