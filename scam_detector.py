"""
Scam Detector - Scam detection logic for ScamShield
Analyzes messages for suspicious patterns and calculates risk scores
"""

import re
from typing import List, Tuple
from dataclasses import dataclass


SUSPICIOUS_PATTERNS = {
    "urgency": [
        "immediately", "urgent", "act now", "within 24 hours",
        "within 48 hours", "right away", "asap", "expires today",
        "time sensitive", "limited time", "hurry", "instant",
        "right now", "today only", "must act now", "deadline",
    ],
    "threats": [
        "account will be closed", "account will be suspended", "legal action",
        "arrest warrant", "police", "court", "lawsuit", "terminate",
        "permanently banned", "frozen", "seize", "confiscate",
        "investigation", "audit", "compliance", "violation",
        "illegal activity", "fraud investigation", "security breach",
    ],
    "financial_pressure": [
        "send money", "transfer money", "wire transfer", "gift card",
        "bitcoin", "cryptocurrency", "paytm", "google pay",
        "phonepe", "pay", "payment", "bank details", "card details",
        "account number", "ifsc code", "neft", "rtgs", "upi",
        "refund", "credit card", "debit card", "₹", "$", "€",
        "cash", "recharge", "fees", "charges",
    ],
    "prize_lottery": [
        "you have won", "congratulations", "prize", "lottery", "winner",
        "jackpot", "award", "reward", "claim prize", "claim now",
        "free", "bonus", "cash prize", "won", "winning",
        "lucky winner", "entitled to",
    ],
    "impersonation": [
        "bank", "police", "government", "amazon", "flipkart", "paytm",
        "google", "microsoft", "apple", "netflix", "myntra",
        "snapdeal", "airtel", "jio", "customer care", "support",
        "official", "authorized", "verified", "security team",
        "irs", "cbi", "income tax", "reserve bank", "rbi",
    ],
}


@dataclass
class AnalysisResult:
    suspicious_phrases: List[Tuple[str, str]]
    entity_count: int
    financial_request_detected: bool
    urgency_score: int
    threat_score: int
    financial_score: int
    prize_score: int
    impersonation_score: int
    total_risk_score: int
    explanation: List[str]


def find_suspicious_phrases(text: str) -> List[Tuple[str, str]]:
    """Find all suspicious phrases in text"""
    text_lower = text.lower()
    found_phrases = []
    
    for category, patterns in SUSPICIOUS_PATTERNS.items():
        for pattern in patterns:
            regex = r'\b' + re.escape(pattern.lower()) + r'\b'
            if re.search(regex, text_lower):
                found_phrases.append((pattern, category))
    
    return found_phrases


def calculate_category_score(phrases: List[Tuple[str, str]], category: str) -> int:
    """Calculate score for a specific category (0-100)"""
    category_phrases = [p for p in phrases if p[1] == category]
    count = len(category_phrases)
    
    if count == 0:
        return 0
    elif count == 1:
        return 25
    elif count == 2:
        return 50
    elif count == 3:
        return 75
    else:
        return 100


def detect_financial_request(text: str) -> bool:
    """Detect if message contains a financial request"""
    financial_keywords = [
        "send", "transfer", "pay", "money", "₹", "$", "rupee", "dollar",
        "account", "card", "upi", "bitcoin", "gift card", "recharge",
        "deposit", "withdraw", "payment", "refund",
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in financial_keywords)


def calculate_entity_risk(entity_count: int) -> int:
    """Calculate risk based on entity count (0-20)"""
    if entity_count <= 3:
        return 0
    elif entity_count <= 6:
        return 5
    elif entity_count <= 10:
        return 10
    elif entity_count <= 15:
        return 15
    else:
        return 20


def analyze_message(text: str, entity_count: int = 0) -> AnalysisResult:
    """Analyze a message for scam indicators"""
    suspicious_phrases = find_suspicious_phrases(text)
    financial_request_detected = detect_financial_request(text)
    
    urgency_score = calculate_category_score(suspicious_phrases, "urgency")
    threat_score = calculate_category_score(suspicious_phrases, "threats")
    financial_score = calculate_category_score(suspicious_phrases, "financial_pressure")
    prize_score = calculate_category_score(suspicious_phrases, "prize_lottery")
    impersonation_score = calculate_category_score(suspicious_phrases, "impersonation")
    
    entity_risk = calculate_entity_risk(entity_count)
    
    pattern_score = (
        urgency_score * 0.2 +
        threat_score * 0.25 +
        financial_score * 0.3 +
        prize_score * 0.15 +
        impersonation_score * 0.1
    )
    
    total_risk = min(100, int(pattern_score + entity_risk))
    
    explanations = []
    
    if total_risk >= 70:
        explanations.append("HIGH RISK: This message exhibits multiple scam characteristics")
    elif total_risk >= 40:
        explanations.append("MODERATE RISK: This message shows some suspicious indicators")
    else:
        explanations.append("LOW RISK: This message appears relatively safe")
    
    if urgency_score >= 50:
        explanations.append(f"Strong urgency indicators detected (score: {urgency_score})")
    if threat_score >= 50:
        explanations.append(f"Threatening language detected (score: {threat_score})")
    if financial_score >= 50:
        explanations.append(f"Financial request/pressure detected (score: {financial_score})")
    if prize_score >= 50:
        explanations.append(f"Prize/lottery scam indicators found (score: {prize_score})")
    if impersonation_score >= 50:
        explanations.append(f"Potential impersonation of trusted entity (score: {impersonation_score})")
    if entity_risk > 10:
        explanations.append(f"High density of entities may indicate data harvesting")
    if financial_request_detected:
        explanations.append("Financial request or payment language detected")
    if not explanations:
        explanations.append("No obvious scam patterns detected")
    
    return AnalysisResult(
        suspicious_phrases=suspicious_phrases,
        entity_count=entity_count,
        financial_request_detected=financial_request_detected,
        urgency_score=urgency_score,
        threat_score=threat_score,
        financial_score=financial_score,
        prize_score=prize_score,
        impersonation_score=impersonation_score,
        total_risk_score=total_risk,
        explanation=explanations
    )


def get_risk_score(analysis: AnalysisResult) -> int:
    """Get the risk score from an analysis result"""
    return analysis.total_risk_score


def get_risk_level(score: int) -> str:
    """Get risk level label from score"""
    if score <= 30:
        return "LOW"
    elif score <= 60:
        return "MEDIUM"
    else:
        return "HIGH"


def get_risk_color(score: int) -> str:
    """Get risk color from score"""
    if score <= 30:
        return "#28A745"
    elif score <= 60:
        return "#FFC107"
    else:
        return "#DC3545"


def highlight_suspicious(text: str, suspicious_phrases: List[Tuple[str, str]]) -> str:
    """Create HTML-highlighted version of text with suspicious phrases marked"""
    if not suspicious_phrases:
        return text
    
    category_colors = {
        "urgency": "#FF6B6B",
        "threats": "#DC3545",
        "financial_pressure": "#F39C12",
        "prize_lottery": "#9B59B6",
        "impersonation": "#3498DB",
    }
    
    highlighted = text
    for phrase, category in suspicious_phrases:
        color = category_colors.get(category, "#808080")
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; color: white; font-weight: bold;">\\g<0></span>',
            highlighted
        )
    
    return highlighted


def get_suspicious_phrases(text: str) -> List[Tuple[str, str]]:
    """Convenience function to get suspicious phrases from text"""
    return find_suspicious_phrases(text)


if __name__ == "__main__":
    test_messages = [
        "URGENT: Your bank account has been compromised. Transfer ₹50,000 immediately to secure account or legal action will be taken.",
        "Congratulations! You've won ₹10,00,000 in the Flipkart lottery. Click https://win.in/claim to claim your prize now!",
        "Dear customer, your Amazon order #12345 will be delivered tomorrow. Track at amazon.com/track",
        "WARNING: Your account will be permanently closed within 24 hours if you don't verify your KYC. Call +91-9876543210 immediately.",
        "Hello, this is from ICICI bank. We need to verify your account details. Please send your card number and OTP to confirm.",
    ]
    
    print("Testing Scam Detector:\n")
    for msg in test_messages:
        print(f"Message: {msg[:80]}...")
        analysis = analyze_message(msg, entity_count=3)
        print(f"Risk Score: {analysis.total_risk_score}/100 ({get_risk_level(analysis.total_risk_score)})")
        print(f"Urgency: {analysis.urgency_score}, Threats: {analysis.threat_score}, Financial: {analysis.financial_score}")
        print(f"Suspicious phrases found: {len(analysis.suspicious_phrases)}")
        for phrase, cat in analysis.suspicious_phrases:
            print(f"  - '{phrase}' ({cat})")
        print()
