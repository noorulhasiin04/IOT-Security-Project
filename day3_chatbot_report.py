"""
Day 3 - Interactive Chatbot Q&A + Weekly PDF Security Report
Project: Generative AI-Enhanced IoT Security Using Deep Neural Networks

Features:
1. chat_about_alert(): Answers user follow-up questions regarding specific alert details.
2. generate_weekly_report(): Aggregates session alert history, generates executive AI summaries,
   and exports a polished PDF report using fpdf2.
"""

import os
import logging
from datetime import date
from dotenv import load_dotenv
from fpdf import FPDF

# Load environment variables
load_dotenv()

# ---------- CONFIGURATION ----------
MODEL_ID = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

genai_client = None
if GEMINI_API_KEY and GEMINI_API_KEY != "PASTE_YOUR_API_KEY_HERE":
    try:
        from google import genai
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logging.warning(f"Could not initialize google-genai client in day3: {e}")


def chat_about_alert(device: str, attack_type: str, confidence: float, question: str, history: list = None) -> str:
    """
    Answers user questions regarding an active security alert with context memory.
    """
    if not question or not question.strip():
        return "Please ask a question regarding the alert."

    # Build prompt context
    history_str = ""
    if history and len(history) > 0:
        history_str = "\nPrevious Conversation History:\n" + "\n".join(
            f"User: {item.get('user')}\nAssistant: {item.get('bot')}" for item in history[-3:]
        )

    prompt = f"""
You are an expert IoT Cybersecurity Assistant helping a user resolve a security alert.

Alert Context:
- Target Device: {device}
- Classification: {attack_type}
- Model Confidence: {confidence:.1f}%
{history_str}

User Question: "{question}"

Instructions:
- Provide a direct, plain-language answer in 2-4 sentences.
- Explain technical terms if used.
- Focus on practical guidance for the non-technical administrator.
"""

    if genai_client is not None:
        try:
            response = genai_client.models.generate_content(
                model=MODEL_ID,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            logging.warning(f"Chatbot API error: {e}")

    # Fallback response engine if offline
    q_lower = question.lower()
    if "dangerous" in q_lower or "risk" in q_lower or "severe" in q_lower:
        return f"This {attack_type} alert on {device} is classified with high risk. An attacker may disrupt network operations or compromise device functions. Immediate isolation is advised."
    elif "fix" in q_lower or "do" in q_lower or "protect" in q_lower or "solve" in q_lower:
        return f"To secure {device} against {attack_type}, disconnect it from public routing, update its credentials, and verify firewall packet filtering rules."
    elif "who" in q_lower or "source" in q_lower or "attacker" in q_lower:
        return f"The attack originates from automated malicious flow traffic directed at {device}. Deep inspection of packet source IPs is recommended to identify the offender."
    else:
        return f"The alert indicates {attack_type} activity targeting {device} with {confidence:.1f}% AI confidence. Review your router logs and isolate affected devices to ensure network safety."


def generate_weekly_report(alerts: list, output_path: str = "weekly_report.pdf") -> str:
    """
    Summarizes alerts and creates a clean PDF security report using fpdf2.
    """
    if not alerts:
        alerts = [
            {"device": "ESP32 Sensor Node", "attack_type": "DDoS", "confidence": 98.5, "severity": "Critical"},
            {"device": "Smart Camera", "attack_type": "Recon", "confidence": 89.2, "severity": "Medium"}
        ]

    # Format alerts text for AI summary
    alerts_text = "\n".join(
        f"- Device: {a['device']} | Attack: {a['attack_type']} | Severity: {a.get('severity', 'High')} | Confidence: {a['confidence']:.1f}%"
        for a in alerts
    )

    summary_text = ""
    prompt = f"""
You are writing a weekly executive IoT security report based on these logged alerts:
{alerts_text}

Provide a concise 4-5 sentence executive summary detailing:
1. Total number of security incidents and affected devices.
2. Most severe threat observed and potential impact.
3. Top 2 concrete recommendations for system hardening next week.
"""

    if genai_client is not None:
        try:
            response = genai_client.models.generate_content(
                model=MODEL_ID,
                contents=prompt
            )
            summary_text = response.text.strip()
        except Exception as e:
            logging.warning(f"Report summary API error: {e}")

    if not summary_text:
        total_count = len(alerts)
        devices_set = set(a["device"] for a in alerts)
        summary_text = (
            f"During the current monitoring period, a total of {total_count} security incident(s) "
            f"were detected across {len(devices_set)} unique device node(s). "
            f"The primary threat activity detected involved automated intrusion traffic targeting device interfaces. "
            f"It is recommended to enforce strict firewall rules, change default passwords, "
            f"and continuously monitor device network traffic."
        )

    # ---------- BUILD PDF DOCUMENT WITH FPDF2 ----------
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title & Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Weekly IoT Security Executive Report", align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated Date: {date.today().isoformat()} | Pipeline Status: Active", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Section 1: Executive Summary
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "1. Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, summary_text)
    pdf.ln(6)

    # Section 2: Logged Security Incidents Table
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "2. Logged Incident History", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Table Headers
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(45, 8, "Device Node", border=1, align="C")
    pdf.cell(45, 8, "Attack Category", border=1, align="C")
    pdf.cell(45, 8, "Severity Level", border=1, align="C")
    pdf.cell(45, 8, "DNN Confidence", border=1, align="C", new_x="LMARGIN", new_y="NEXT")

    # Table Rows
    pdf.set_font("Helvetica", "", 10)
    for a in alerts:
        pdf.cell(45, 7, str(a.get("device", "N/A"))[:20], border=1)
        pdf.cell(45, 7, str(a.get("attack_type", "N/A"))[:20], border=1)
        pdf.cell(45, 7, str(a.get("severity", "Medium")), border=1)
        pdf.cell(45, 7, f"{a.get('confidence', 0.0):.1f}%", border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)

    # Section 3: Mitigation Recommendations
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "3. System Recommendations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    recs = [
        "1. Isolate compromised IoT nodes onto a dedicated VLAN network segment.",
        "2. Update all network gateway firmware and enforce WPA3/TLS 1.3 encryption.",
        "3. Review automated alert logs daily to detect early reconnaissance scanning."
    ]
    for r in recs:
        pdf.multi_cell(0, 6, r)
        pdf.ln(1)

    # Save PDF
    pdf.output(output_path)
    return output_path


# ---------- QUICK TEST ----------
if __name__ == "__main__":
    print("Testing chat_about_alert()...")
    chat_resp = chat_about_alert("ESP32 Sensor Node", "DDoS", 98.5, "How dangerous is this attack?")
    print("Chat response:\n", chat_resp)

    print("\nTesting generate_weekly_report()...")
    test_alerts = [
        {"device": "ESP32 Sensor Node", "attack_type": "DDoS", "confidence": 98.5, "severity": "Critical"},
        {"device": "Smart Camera", "attack_type": "Recon", "confidence": 88.0, "severity": "Medium"}
    ]
    pdf_file = generate_weekly_report(test_alerts)
    print(f"Report generated successfully: {pdf_file}")
