"""
ScamShield - AI Scam Message & Phishing Language Analyzer
Main Streamlit Application
"""

import streamlit as st
import spacy
from PIL import Image
import io
import base64

from ner_utils import extract_entities, load_model, get_entity_counts, highlight_entities_in_text
from scam_detector import (
    analyze_message, get_risk_score, get_risk_level, get_risk_color,
    highlight_suspicious, get_suspicious_phrases
)
from evaluator import evaluate_ner, get_test_dataset, get_confusion_data

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


st.set_page_config(
    page_title="ScamShield - AI Scam Analyzer",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="collapsed"
)


with open("styles.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_nlp_model():
    """Load and cache the spaCy model"""
    return load_model()


def load_image_base64(path):
    """Load image and return base64 string"""
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None


def create_gauge_chart(score, title="Risk Score"):
    """Create a gauge chart for risk score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "darkgray"},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#10B981'},
                {'range': [30, 60], 'color': '#F59E0B'},
                {'range': [60, 100], 'color': '#EF4444'},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        },
        title={'text': title, 'font': {'size': 16, 'color': '#F8FAFC'}},
        number={'font': {'size': 48, 'color': '#F8FAFC'}},
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#F8FAFC'},
        height=250,
        margin={'l': 20, 'r': 20, 't': 50, 'b': 20},
    )
    
    return fig


def create_entity_pie_chart(entity_counts):
    """Create pie chart for entity distribution"""
    if not entity_counts:
        return None
    
    df = pd.DataFrame(list(entity_counts.items()), columns=['Entity Type', 'Count'])
    
    colors = {
        'PERSON': '#FF6B6B', 'ORG': '#4ECDC4', 'GPE': '#45B7D1',
        'DATE': '#96CEB4', 'TIME': '#FFEAA7', 'MONEY': '#DDA0DD',
        'PHONE': '#FF8C00', 'EMAIL': '#9B59B6', 'URL': '#3498DB',
        'UPI': '#E74C3C', 'CURRENCY': '#F39C12'
    }
    
    fig = px.pie(
        df, 
        values='Count', 
        names='Entity Type',
        title='Entity Distribution',
        color='Entity Type',
        color_discrete_map=colors,
        hole=0.4
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#F8FAFC'},
        legend={'font': {'color': '#F8FAFC'}},
        title={'font': {'color': '#F8FAFC'}},
    )
    
    return fig


def create_detection_bar_chart(urgency, threats, financial, prize, impersonation):
    """Create bar chart for detection categories"""
    categories = ['Urgency', 'Threats', 'Financial', 'Prize/Lottery', 'Impersonation']
    scores = [urgency, threats, financial, prize, impersonation]
    colors = ['#FF6B6B', '#DC3545', '#F39C12', '#9B59B6', '#3498DB']
    
    fig = px.bar(
        x=categories,
        y=scores,
        title='Detection Category Scores',
        color=colors,
        color_discrete_sequence=colors,
        labels={'x': 'Category', 'y': 'Score'}
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#F8FAFC'},
        xaxis={'color': '#F8FAFC'},
        yaxis={'color': '#F8FAFC', 'range': [0, 100]},
        title={'font': {'color': '#F8FAFC'}},
        showlegend=False,
    )
    
    return fig


def main():
    # Header with icon and title centered
    st.markdown("""
        <div class="main-header">
            <div class="header-content">
                <span class="shield-icon">
                    <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2L4 5V11.09C4 16.14 7.41 20.85 12 22C16.59 20.85 20 16.14 20 11.09V5L12 2Z" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M12 8V12M12 16H12.01" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </span>
                <div class="title-block">
                    <h1>ScamShield</h1>
                    <p>AI-Powered Scam Message & Phishing Language Analyzer</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    tab_names = ["Message Analysis", "NER Evaluation", "Dashboard", "About"]
    
    tab1, tab2, tab3, tab4 = st.tabs(tab_names)
    
    with tab1:
        st.markdown("### Analyze Message")
        st.markdown("Enter an SMS, Email, or WhatsApp message to analyze for potential scam indicators.")
        
        message_input = st.text_area(
            "Enter message to analyze:",
            placeholder="e.g., URGENT: Your bank account has been compromised. Transfer ₹50000 immediately...",
            height=150,
            key="message_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button("Analyze Message", use_container_width=True)
        
        if analyze_button and message_input:
            with st.spinner("Analyzing message..."):
                nlp = get_nlp_model()
                entities = extract_entities(message_input, nlp)
                analysis = analyze_message(message_input, len(entities))
            
            risk_score = analysis.total_risk_score
            risk_level = get_risk_level(risk_score)
            risk_color = get_risk_color(risk_score)
            
            st.markdown("---")
            st.markdown("### Analysis Results")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.plotly_chart(create_gauge_chart(risk_score), use_container_width=True)
            
            with metric_col2:
                st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid {risk_color};">
                        <div class="metric-value" style="color: {risk_color};">{risk_level}</div>
                        <div class="metric-label">Risk Level</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(entities)}</div>
                        <div class="metric-label">Entities Found</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with metric_col4:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(analysis.suspicious_phrases)}</div>
                        <div class="metric-label">Suspicious Phrases</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.plotly_chart(
                    create_detection_bar_chart(
                        analysis.urgency_score,
                        analysis.threat_score,
                        analysis.financial_score,
                        analysis.prize_score,
                        analysis.impersonation_score
                    ),
                    use_container_width=True
                )
            
            with viz_col2:
                entity_counts = get_entity_counts(entities)
                if entity_counts:
                    st.plotly_chart(create_entity_pie_chart(entity_counts), use_container_width=True)
                else:
                    st.info("No entities found in the message.")
            
            st.markdown("---")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.markdown("#### Extracted Entities")
                if entities:
                    entity_df = pd.DataFrame([
                        {"Entity": e["text"], "Type": e["label"], "Position": f"{e['start']}-{e['end']}"}
                        for e in entities
                    ])
                    st.dataframe(entity_df, use_container_width=True)
                else:
                    st.info("No entities found.")
            
            with result_col2:
                st.markdown("#### Suspicious Phrases")
                if analysis.suspicious_phrases:
                    phrase_df = pd.DataFrame([
                        {"Phrase": phrase, "Category": cat.capitalize()}
                        for phrase, cat in analysis.suspicious_phrases
                    ])
                    st.dataframe(phrase_df, use_container_width=True)
                else:
                    st.info("No suspicious phrases detected.")
            
            st.markdown("---")
            st.markdown("#### Analysis Explanation")
            
            for exp in analysis.explanation:
                if "HIGH RISK" in exp:
                    st.error(exp)
                elif "MODERATE RISK" in exp:
                    st.warning(exp)
                else:
                    st.success(exp)
            
            st.markdown("---")
            
            highlighted_text = highlight_suspicious(message_input, analysis.suspicious_phrases)
            st.markdown("#### Highlighted Message")
            st.markdown(
                f'<div style="background-color: #1E293B; padding: 20px; border-radius: 12px; border: 1px solid #475569;">{highlighted_text}</div>',
                unsafe_allow_html=True
            )
    
    with tab2:
        st.markdown("### NER Evaluation")
        st.markdown("Evaluate the Named Entity Recognition model's performance on the test dataset.")
        
        eval_button = st.button("Run Evaluation", use_container_width=True)
        
        if eval_button:
            with st.spinner("Evaluating NER model..."):
                try:
                    nlp = get_nlp_model()
                    results = evaluate_ner(nlp)
                    
                    st.markdown("#### Performance Metrics")
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric("Macro Precision", f"{results['macro_precision']:.2%}")
                    with metrics_col2:
                        st.metric("Macro Recall", f"{results['macro_recall']:.2%}")
                    with metrics_col3:
                        st.metric("Macro F1-Score", f"{results['macro_f1']:.2%}")
                    
                    st.markdown("#### Entity-Level Performance")
                    st.dataframe(results['metrics_df'], use_container_width=True)
                    
                    st.markdown(f"""
                        <div class="info-box">
                            <strong>Evaluation Summary:</strong><br>
                            Total Test Messages: {results['total_messages']}<br>
                            Total True Entities: {results['total_true_entities']}<br>
                            Total Predicted Entities: {results['total_pred_entities']}
                        </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
                    st.info("Make sure spaCy model and all dependencies are installed.")
        
        st.markdown("---")
        st.markdown("#### Test Dataset Sample")
        dataset = get_test_dataset()[:10]
        for i, msg in enumerate(dataset):
            with st.expander(f"Message {i+1}: {msg.text[:60]}..."):
                st.text(f"Text: {msg.text}")
                st.text(f"Entities: {[(e.text, e.label) for e in msg.entities]}")
    
    with tab3:
        st.markdown("### Dashboard")
        st.markdown("Overview of ScamShield analytics and statistics.")
        
        sample_messages = [
            "URGENT: Your HDFC account has been compromised. Transfer ₹50000 immediately.",
            "Congratulations! You've won ₹10,00,000 in the Flipkart lottery.",
            "Dear customer, your Amazon order #12345 will be delivered tomorrow.",
            "WARNING: Your account will be closed within 24 hours if you don't verify.",
            "Hello, this is from ICICI bank. We need to verify your account details.",
            "You've received ₹1000 cashback. Claim within 24 hours!",
        ]
        
        all_entities = []
        all_scores = []
        all_categories = {"urgency": 0, "threats": 0, "financial": 0, "prize": 0, "impersonation": 0}
        
        nlp = get_nlp_model()
        for msg in sample_messages:
            entities = extract_entities(msg, nlp)
            analysis = analyze_message(msg, len(entities))
            all_entities.extend(entities)
            all_scores.append(analysis.total_risk_score)
            all_categories["urgency"] += analysis.urgency_score
            all_categories["threats"] += analysis.threat_score
            all_categories["financial"] += analysis.financial_score
            all_categories["prize"] += analysis.prize_score
            all_categories["impersonation"] += analysis.impersonation_score
        
        dash_col1, dash_col2, dash_col3, dash_col4 = st.columns(4)
        
        with dash_col1:
            st.metric("Messages Analyzed", len(sample_messages))
        with dash_col2:
            st.metric("Total Entities Found", len(all_entities))
        with dash_col3:
            st.metric("Avg Risk Score", f"{sum(all_scores)/len(all_scores):.1f}")
        with dash_col4:
            high_risk = sum(1 for s in all_scores if s > 60)
            st.metric("High Risk Messages", high_risk)
        
        st.markdown("#### Category Score Distribution")
        
        cat_df = pd.DataFrame([all_categories])
        st.bar_chart(cat_df)
        
        st.markdown("#### Risk Score Distribution")
        score_df = pd.DataFrame({"Score": all_scores})
        st.line_chart(score_df)
        
        entity_counts = get_entity_counts(all_entities)
        if entity_counts:
            st.plotly_chart(create_entity_pie_chart(entity_counts), use_container_width=True)
    
    with tab4:
        st.markdown("### About ScamShield")
        
        st.markdown("""
            <div class="info-box">
                <h3>ScamShield</h3>
                <p><strong>Version:</strong> 1.0.0</p>
                <p><strong>Purpose:</strong> AI-powered scam message and phishing language analyzer</p>
                <p>ScamShield uses Natural Language Processing (NLP) and Machine Learning to analyze 
                text messages and identify potential scam indicators.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Features")
        
        features = [
            ("Named Entity Recognition", "Extracts entities like phone numbers, emails, URLs, UPI IDs, and currency amounts using spaCy"),
            ("Suspicious Pattern Detection", "Identifies urgency, threats, financial pressure, prize/lottery, and impersonation patterns"),
            ("Risk Scoring", "Calculates a 0-100 risk score based on multiple indicators"),
            ("Visual Analysis", "Interactive charts and graphs for better understanding"),
            ("Multi-format Support", "Analyze SMS, Emails, and WhatsApp messages"),
        ]
        
        for feature, desc in features:
            st.markdown(f"**{feature}**: {desc}")
        
        st.markdown("#### Entity Types Detected")
        
        entity_types = {
            "Built-in Entities": ["PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY"],
            "Custom Entities": ["PHONE", "EMAIL", "URL", "UPI", "CURRENCY"]
        }
        
        for category, types in entity_types.items():
            st.markdown(f"**{category}**: {', '.join(types)}")
    
    st.markdown("""
        <div class="footer">
            <p>ScamShield - AI Scam Analyzer | Built with Streamlit, spaCy & Plotly</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()