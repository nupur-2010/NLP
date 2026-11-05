"""
Evaluator - NER Evaluation for ScamShield
Evaluates Named Entity Recognition performance with Precision, Recall, F1
Uses both spaCy built-in NER and regex-based custom entity extraction
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import spacy
import pandas as pd
import re
from collections import defaultdict


@dataclass
class EntityAnnotation:
    text: str
    label: str
    start: int
    end: int


@dataclass
class AnnotatedMessage:
    text: str
    entities: List[EntityAnnotation]


# Regex patterns for custom entities
REGEX_PATTERNS = [
    # Phone numbers
    (r"\+91[\s\-]?\d{10}", "PHONE"),
    (r"\b\d{10}\b", "PHONE"),
    (r"\+\d{1,3}[\s\-]?\d{6,12}", "PHONE"),
    (r"\b\d{3}[\-\s]?\d{3}[\-\s]?\d{4}\b", "PHONE"),
    (r"\b\d{4}[\-\s]?\d{3}[\-\s]?\d{3}\b", "PHONE"),
    (r"\b1800[\-\s]?\d{3}[\-\s]?\d{3}\b", "PHONE"),
    # Email
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "EMAIL"),
    # URLs
    (r"https?://[^\s]+", "URL"),
    (r"www\.[^\s]+", "URL"),
    # UPI IDs
    (r"[a-zA-Z0-9._-]+@[a-zA-Z]+", "UPI"),
    # Currency
    (r"₹\s?\d+[\d,]*(\.\d{2})?", "CURRENCY"),
    (r"Rs\.?\s?\d+[\d,]*(\.\d{2})?", "CURRENCY"),
    (r"\$\s?\d+[\d,]*(\.\d{2})?", "CURRENCY"),
    (r"USD\s?\d+[\d,]*(\.\d{2})?", "CURRENCY"),
    (r"€\s?\d+[\d,]*(\.\d{2})?", "CURRENCY"),
]


def extract_regex_entities(text: str) -> List[Dict]:
    """Extract custom entities using regex patterns"""
    entities = []
    for pattern, label in REGEX_PATTERNS:
        for match in re.finditer(pattern, text):
            entities.append({
                "text": match.group(),
                "label": label,
                "start": match.start(),
                "end": match.end()
            })
    return entities


def get_test_dataset() -> List[AnnotatedMessage]:
    """Get test dataset of annotated scam messages"""
    return [
        AnnotatedMessage(
            text="URGENT: Your HDFC bank account has been compromised. Transfer ₹50000 immediately to secure account or legal action will be taken. Call 9876543210.",
            entities=[
                EntityAnnotation("HDFC", "ORG", 18, 22),
                EntityAnnotation("₹50000", "CURRENCY", 68, 76),
                EntityAnnotation("9876543210", "PHONE", 147, 157),
            ]
        ),
        AnnotatedMessage(
            text="Congratulations! You've won ₹10,00,000 in the Flipkart lottery. Click https://claim.flipkart.com to claim your prize now!",
            entities=[
                EntityAnnotation("₹10,00,000", "CURRENCY", 21, 31),
                EntityAnnotation("Flipkart", "ORG", 36, 45),
                EntityAnnotation("https://claim.flipkart.com", "URL", 71, 96),
            ]
        ),
        AnnotatedMessage(
            text="Your ICICI account will be permanently closed within 24 hours if you don't verify your KYC. Call +91-9876543210 immediately.",
            entities=[
                EntityAnnotation("ICICI", "ORG", 9, 14),
                EntityAnnotation("24 hours", "TIME", 61, 70),
                EntityAnnotation("+91-9876543210", "PHONE", 101, 115),
            ]
        ),
        AnnotatedMessage(
            text="Dear customer, this is from Amazon security team. Your account has been hacked. Send ₹2000 to name@upi to secure your account immediately.",
            entities=[
                EntityAnnotation("Amazon", "ORG", 25, 31),
                EntityAnnotation("₹2000", "CURRENCY", 70, 76),
                EntityAnnotation("name@upi", "UPI", 80, 89),
            ]
        ),
        AnnotatedMessage(
            text="You have won a gift card worth $500 from Google! Claim your prize at www.google-reward.com before it expires today.",
            entities=[
                EntityAnnotation("$500", "CURRENCY", 20, 25),
                EntityAnnotation("Google", "ORG", 34, 40),
                EntityAnnotation("www.google-reward.com", "URL", 56, 78),
                EntityAnnotation("today", "DATE", 95, 100),
            ]
        ),
        AnnotatedMessage(
            text="ALERT: Your SBI account has been frozen due to suspicious activity. To unfreeze, send ₹10000 to sbi@upi within 24 hours or face legal action.",
            entities=[
                EntityAnnotation("SBI", "ORG", 9, 12),
                EntityAnnotation("₹10000", "CURRENCY", 62, 70),
                EntityAnnotation("sbi@upi", "UPI", 74, 82),
                EntityAnnotation("24 hours", "TIME", 87, 96),
            ]
        ),
        AnnotatedMessage(
            text="Congratulations dear customer! You've won ₹5,00,000 in the National Lottery. Contact claim@lottery.com to claim your prize.",
            entities=[
                EntityAnnotation("₹5,00,000", "CURRENCY", 28, 38),
                EntityAnnotation("National Lottery", "ORG", 43, 58),
                EntityAnnotation("claim@lottery.com", "EMAIL", 74, 92),
            ]
        ),
        AnnotatedMessage(
            text="URGENT: Your Paytm account has been suspended. Verify your details by sending ₹500 to verify@paytm within 1 hour or lose your account forever.",
            entities=[
                EntityAnnotation("Paytm", "ORG", 18, 23),
                EntityAnnotation("₹500", "CURRENCY", 62, 66),
                EntityAnnotation("verify@paytm", "UPI", 80, 93),
                EntityAnnotation("1 hour", "TIME", 103, 109),
            ]
        ),
        AnnotatedMessage(
            text="This is from the Income Tax Department. You have a pending refund of ₹45000. Click https://incometax.gov.in/refund to claim it immediately.",
            entities=[
                EntityAnnotation("Income Tax Department", "ORG", 12, 34),
                EntityAnnotation("₹45000", "CURRENCY", 60, 68),
                EntityAnnotation("https://incometax.gov.in/refund", "URL", 79, 111),
            ]
        ),
        AnnotatedMessage(
            text="Your bank account is at risk! Send your ATM card details to secure@bank.com immediately or ₹20000 will be deducted. Call 9876543210.",
            entities=[
                EntityAnnotation("₹20000", "CURRENCY", 90, 98),
                EntityAnnotation("secure@bank.com", "EMAIL", 47, 62),
                EntityAnnotation("9876543210", "PHONE", 131, 141),
            ]
        ),
        AnnotatedMessage(
            text="Hi Rahul, can we meet tomorrow at 3 PM for the project discussion? Let me know your availability.",
            entities=[
                EntityAnnotation("Rahul", "PERSON", 3, 8),
                EntityAnnotation("tomorrow", "DATE", 21, 30),
                EntityAnnotation("3 PM", "TIME", 31, 35),
            ]
        ),
        AnnotatedMessage(
            text="The meeting has been scheduled for Monday at 10 AM in Conference Room A. Please confirm your attendance.",
            entities=[
                EntityAnnotation("Monday", "DATE", 32, 38),
                EntityAnnotation("10 AM", "TIME", 43, 48),
            ]
        ),
        AnnotatedMessage(
            text="Thanks for shopping at Myntra! Your order worth ₹2500 will be delivered by Saturday. Track at myntra.com.",
            entities=[
                EntityAnnotation("Myntra", "ORG", 24, 30),
                EntityAnnotation("₹2500", "CURRENCY", 41, 47),
                EntityAnnotation("Saturday", "DATE", 73, 81),
                EntityAnnotation("myntra.com", "URL", 92, 103),
            ]
        ),
        AnnotatedMessage(
            text="Your Swiggy order #SW12345 is being prepared. Expected delivery: 45 minutes. Track at swiggy.in.",
            entities=[
                EntityAnnotation("Swiggy", "ORG", 5, 11),
                EntityAnnotation("SW12345", "ORG", 17, 24),
                EntityAnnotation("45 minutes", "TIME", 59, 70),
                EntityAnnotation("swiggy.in", "URL", 82, 92),
            ]
        ),
        AnnotatedMessage(
            text="Reminder: Your dentist appointment is on 20th April at 2:30 PM at Apollo Hospitals. Please arrive 10 minutes early.",
            entities=[
                EntityAnnotation("20th April", "DATE", 32, 42),
                EntityAnnotation("2:30 PM", "TIME", 47, 54),
                EntityAnnotation("Apollo Hospitals", "ORG", 59, 75),
            ]
        ),
        AnnotatedMessage(
            text="Your gas cylinder booking for Bharat Gas is confirmed. Delivery expected within 3 days. Call 9876543210 for details.",
            entities=[
                EntityAnnotation("Bharat Gas", "ORG", 31, 41),
                EntityAnnotation("3 days", "TIME", 70, 77),
                EntityAnnotation("9876543210", "PHONE", 95, 105),
            ]
        ),
        AnnotatedMessage(
            text="Your LIC insurance premium of ₹5000 is due on 15th March. Pay at lic.co.in/pay to avoid penalty.",
            entities=[
                EntityAnnotation("LIC", "ORG", 5, 8),
                EntityAnnotation("₹5000", "CURRENCY", 33, 39),
                EntityAnnotation("15th March", "DATE", 52, 63),
                EntityAnnotation("lic.co.in/pay", "URL", 80, 92),
            ]
        ),
        AnnotatedMessage(
            text="Your Jio postpaid bill for March is ₹1500. Pay now at jio.com/pay or via JioPay app.",
            entities=[
                EntityAnnotation("Jio", "ORG", 5, 8),
                EntityAnnotation("March", "DATE", 25, 30),
                EntityAnnotation("₹1500", "CURRENCY", 36, 42),
                EntityAnnotation("jio.com/pay", "URL", 57, 69),
            ]
        ),
        AnnotatedMessage(
            text="Dear student, your exam result for CBSE Class 12 has been declared. Check at cbse.gov.in/results.",
            entities=[
                EntityAnnotation("CBSE", "ORG", 33, 37),
                EntityAnnotation("Class 12", "ORG", 38, 46),
                EntityAnnotation("cbse.gov.in/results", "URL", 75, 93),
            ]
        ),
        AnnotatedMessage(
            text="Congratulations! You've won a free recharge of ₹300. Claim it within 24 hours.",
            entities=[
                EntityAnnotation("₹300", "CURRENCY", 31, 36),
                EntityAnnotation("24 hours", "TIME", 59, 68),
            ]
        ),
    ]


def evaluate_ner(nlp=None) -> Dict:
    """
    Evaluate NER performance using the test dataset
    Combines spaCy built-in NER with regex-based custom entity extraction
    for comprehensive entity recognition evaluation
    """
    if nlp is None:
        from ner_utils import load_model
        nlp = load_model()
    
    dataset = get_test_dataset()
    
    # Count TP, FP, FN per entity type
    tp_by_type = defaultdict(int)
    fp_by_type = defaultdict(int)
    fn_by_type = defaultdict(int)
    
    for msg in dataset:
        doc = nlp(msg.text)
        
        # Get spaCy predictions (built-in entities only)
        spacy_preds = {}
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'TIME', 'MONEY', 'CARDINAL', 'EVENT', 'FAC', 'LANGUAGE', 'LAW', 'LOC', 'NORP', 'ORDINAL', 'PERCENT', 'PRODUCT', 'QUANTITY', 'WORK_OF_ART']:
                key = (ent.text.strip(), ent.start_char, ent.end_char, ent.label_)
                spacy_preds[key] = ent.label_
        
        # Get regex predictions (custom entities)
        regex_preds = {}
        for ent in extract_regex_entities(msg.text):
            key = (ent["text"].strip(), ent["start"], ent["end"], ent["label"])
            regex_preds[key] = ent["label"]
        
        # Combine all predictions (regex takes priority for custom types)
        all_preds = {**spacy_preds, **regex_preds}
        
        # Build list of predicted entities (text, start, end, label)
        pred_ents = list(all_preds.items())
        
        # Build list of true entities
        true_ents = [(e.text, e.start, e.end, e.label) for e in msg.entities]
        
        # Track which predictions have been matched
        matched_preds = set()
        matched_trues = set()
        
        # Match predictions to ground truth (by text + label + position)
        for i, (t_text, t_start, t_end, t_label) in enumerate(true_ents):
            for j, (key, p_label) in enumerate(pred_ents):
                if j in matched_preds:
                    continue
                # Match if text and label are the same
                if t_text.strip() == key[0] and t_label == p_label:
                    tp_by_type[t_label] += 1
                    matched_preds.add(j)
                    matched_trues.add(i)
                    break
        
        # Count remaining predictions as false positives
        for j, (key, p_label) in enumerate(pred_ents):
            if j not in matched_preds:
                fp_by_type[p_label] += 1
        
        # Count unmatched ground truths as false negatives
        for i, (_, _, _, t_label) in enumerate(true_ents):
            if i not in matched_trues:
                fn_by_type[t_label] += 1
    
    # Get all entity types found in either true or predicted
    all_types = set(tp_by_type.keys()) | set(fp_by_type.keys()) | set(fn_by_type.keys())
    
    # Calculate metrics per entity type
    metrics_data = []
    for ent_type in sorted(all_types):
        tp = tp_by_type[ent_type]
        fp = fp_by_type[ent_type]
        fn = fn_by_type[ent_type]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn
        
        metrics_data.append({
            'Entity Type': ent_type,
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4),
            'Support': support
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Calculate macro averages
    total_tp = sum(tp_by_type.values())
    total_fp = sum(fp_by_type.values())
    total_fn = sum(fn_by_type.values())
    
    macro_precision = metrics_df['Precision'].mean() if len(metrics_df) > 0 else 0
    macro_recall = metrics_df['Recall'].mean() if len(metrics_df) > 0 else 0
    macro_f1 = metrics_df['F1-Score'].mean() if len(metrics_df) > 0 else 0
    
    return {
        'metrics_df': metrics_df,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'total_true_entities': total_tp + total_fn,
        'total_pred_entities': total_tp + total_fp,
        'total_messages': len(dataset),
    }


def get_confusion_data(nlp=None) -> pd.DataFrame:
    """Get confusion matrix data for entity types"""
    if nlp is None:
        from ner_utils import load_model
        nlp = load_model()
    
    dataset = get_test_dataset()
    
    true_labels = []
    pred_labels = []
    
    for msg in dataset:
        doc = nlp(msg.text)
        
        pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        
        for ann in msg.entities:
            matched = False
            for pred_start, pred_end, pred_label in pred_entities:
                if pred_start == ann.start and pred_end == ann.end:
                    true_labels.append(ann.label)
                    pred_labels.append(pred_label)
                    matched = True
                    break
            if not matched:
                true_labels.append(ann.label)
                pred_labels.append('MISSED')
    
    all_labels = sorted(set(true_labels + pred_labels))
    
    conf_data = []
    for true_label in all_labels:
        row = {'Actual': true_label}
        for pred_label in all_labels:
            count = sum(1 for t, p in zip(true_labels, pred_labels) if t == true_label and p == pred_label)
            row[pred_label] = count
        conf_data.append(row)
    
    return pd.DataFrame(conf_data)


if __name__ == "__main__":
    from ner_utils import load_model
    
    print("Loading model...")
    nlp = load_model()
    
    print("Evaluating NER...")
    results = evaluate_ner(nlp)
    
    print("\n=== NER Evaluation Results ===\n")
    print(results['metrics_df'].to_string(index=False))
    print(f"\nMacro Avg - Precision: {results['macro_precision']:.2f}, Recall: {results['macro_recall']:.2f}, F1: {results['macro_f1']:.2f}")
