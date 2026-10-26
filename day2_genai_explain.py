"""
Day 2 - Generative AI Explanation Layer
Project: Generative AI-Enhanced IoT Security Using Deep Neural Networks

Upgraded to official Google Gemini SDK (`google-genai`).
Translates technical DNN classification output (Device, Attack Type, Confidence %)
into plain-language explanations, severity levels (Low, Medium, High, Critical),
and 3 prioritized actionable security recommendations.

Features:
- Configurable model identifier constant (default: gemini-2.5-flash).
- API key loading via python-dotenv (.env) or GEMINI_API_KEY environment variable.
- Validated JSON response parsing with automatic single retry.
- Offline rule-based fallback engine so UI demos never crash.
"""

import os
import json
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------- CONFIGURATION ----------
MODEL_ID = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Try initializing official google-genai client
genai_client = None
if GEMINI_API_KEY and GEMINI_API_KEY != "PASTE_YOUR_API_KEY_HERE":
    try:
        from google import genai
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logging.warning(f"Could not initialize google-genai client: {e}")


def get_rule_based_fallback(device: str, attack_type: str, confidence: float) -> dict:
    """
    Offline fallback engine returning clear security explanations when API is unavailable.
    """
    attack_norm = attack_type.strip().lower()
    
    if "benign" in attack_norm:
        return {
            "explanation": f"Normal operational traffic observed from {device}. No malicious anomalies detected.",
            "severity": "Low",
            "recommendations": [
                "No immediate intervention required.",
                "Maintain standard network monitoring.",
                "Keep device firmware updated."
            ]
        }
    elif "ddos" in attack_norm:
        return {
            "explanation": f"Distributed Denial of Service (DDoS) flood targeting {device}. High volume traffic may overwhelm network capacity.",
            "severity": "Critical",
            "recommendations": [
                "Isolate device or restrict incoming rate limits on router.",
                "Enable gateway DDoS protection filters.",
                "Audit traffic sources for automated bot IPs."
            ]
        }
    elif "dos" in attack_norm:
        return {
            "explanation": f"Denial of Service (DoS) attack detected on {device}. Device responsiveness may degrade.",
            "severity": "High",
            "recommendations": [
                "Block offending IP address on firewall.",
                "Restart device network interface if unresponsive.",
                "Enforce connection rate limits."
            ]
        }
    elif "recon" in attack_norm or "scan" in attack_norm:
        return {
            "explanation": f"Reconnaissance scanning detected probing {device} for open ports or system vulnerabilities.",
            "severity": "Medium",
            "recommendations": [
                "Disable unused network services and ports.",
                "Disable ICMP ping responses if unnecessary.",
                "Review firewall logging rules."
            ]
        }
    elif "spoof" in attack_norm:
        return {
            "explanation": f"Spoofing attack detected attempting to forge network identity or manipulate ARP/DNS tables for {device}.",
            "severity": "High",
            "recommendations": [
                "Enable static ARP binding on key gateway nodes.",
                "Inspect DNS server configuration for unauthorized records.",
                "Use encrypted protocols (HTTPS/SSH) exclusively."
            ]
        }
    elif "mirai" in attack_norm:
        return {
            "explanation": f"Mirai botnet malware signature detected targeting {device} via default credential exploits.",
            "severity": "Critical",
            "recommendations": [
                "Change default admin and SSH passwords immediately.",
                "Disable remote Telnet and HTTP management ports.",
                "Reflash device with verified official vendor firmware."
            ]
        }
    elif "web" in attack_norm:
        return {
            "explanation": f"Web application attack attempt (SQLi/XSS/Command Injection) directed at {device} web management interface.",
            "severity": "High",
            "recommendations": [
                "Sanitize and validate all web form input fields.",
                "Update web server component packages.",
                "Deploy Web Application Firewall (WAF) rules."
            ]
        }
    elif "brute" in attack_norm:
        return {
            "explanation": f"Automated password brute-force attempts detected targeting login credentials on {device}.",
            "severity": "High",
            "recommendations": [
                "Enforce account lockout after 3 failed login attempts.",
                "Enforce strong password policy or SSH key authentication.",
                "Restrict login access to trusted internal IP subnets."
            ]
        }
    else:
        return {
            "explanation": f"Potential security anomaly detected on {device} ({attack_type} with {confidence:.1f}% confidence).",
            "severity": "Medium",
            "recommendations": [
                "Review detailed packet logs for suspicious behavior.",
                "Isolate device for diagnostic evaluation.",
                "Ensure security patches are up to date."
            ]
        }


def explain_attack(device: str, attack_type: str, confidence: float) -> dict:
    """
    Sends DNN alert context to Gemini API and returns validated structured JSON explanation.
    Falls back to expert rules if API key is missing or call fails.
    """
    # If client is not configured, immediately return offline fallback
    if genai_client is None:
        return get_rule_based_fallback(device, attack_type, confidence)

    prompt = f"""
You are an expert cybersecurity assistant explaining IoT security alerts to a non-technical administrator.

Alert details:
- Affected Device: {device}
- Classification: {attack_type}
- Model Confidence: {confidence:.1f}%

Respond ONLY in valid raw JSON with strictly these keys:
{{
  "explanation": "1-2 concise, clear sentences describing what this alert means for the device",
  "severity": "One of: Low, Medium, High, Critical",
  "recommendations": ["action item 1", "action item 2", "action item 3"]
}}
Do not add markdown formatting or extra text outside JSON.
"""

    from google.genai import types

    for attempt in range(2):
        try:
            response = genai_client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                )
            )
            text = response.text.strip()
            
            # Clean possible markdown block
            if text.startswith("```"):
                text = text.strip("`").replace("json", "", 1).strip()

            parsed = json.loads(text)
            
            # Validate required keys
            if "explanation" in parsed and "severity" in parsed and "recommendations" in parsed:
                if len(parsed["recommendations"]) >= 1:
                    return parsed
        except Exception as e:
            logging.warning(f"Gemini API attempt {attempt+1} failed: {e}")
            
    # Fallback if API or parsing fails twice
    return get_rule_based_fallback(device, attack_type, confidence)


# ---------- QUICK TEST ----------
if __name__ == "__main__":
    print("Testing explain_attack()...")
    test_result = explain_attack("Smart Camera", "DDoS", 98.5)
    print(json.dumps(test_result, indent=2))
