import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, session, render_template, redirect, url_for, send_file
from flask_cors import CORS
import requests
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from io import BytesIO
import io

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app, supports_credentials=True)  # Enable CORS with credentials
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your_secret_key_here')  # Use environment variable or default

# -----------------------
# Initialize API Keys
# -----------------------
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

print(f"🔑 API Configuration: OpenAI={bool(OPENAI_API_KEY)}, Gemini={bool(GEMINI_API_KEY)}")
if not OPENAI_API_KEY and not GEMINI_API_KEY:
    print("⚠️  No API keys configured. Using rule-based justification fallback.")

# --- MySQL Database Connection ---
db = mysql.connector.connect(
    host=os.environ.get('MYSQL_HOST', 'localhost'),
    user=os.environ.get('MYSQL_USER', 'root'),
    password=os.environ.get('MYSQL_PASSWORD', ''),
    database=os.environ.get('MYSQL_DATABASE', 'insurance_db'),
    auth_plugin='mysql_native_password'
)
cursor = db.cursor(dictionary=True)

# -----------------------
# Initialize Models and Vector Database
# -----------------------
RF_MODEL = None
LE_SEX = None
LE_SMOKER = None
LE_REGION = None
vector_db = None
embedding_model = None

def load_models():
    """Load trained model, encoders, and vector database"""
    global RF_MODEL, LE_SEX, LE_SMOKER, LE_REGION, vector_db, embedding_model
    
    try:
        # Load Random Forest model and encoders
        model_path = os.path.join(os.path.dirname(__file__), "insurance_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
                RF_MODEL = model_data.get("model")
                LE_SEX = model_data.get("sex_encoder")
                LE_SMOKER = model_data.get("smoker_encoder")
                LE_REGION = model_data.get("region_encoder")
                print("✅ RF Model and encoders loaded successfully")
        else:
            print(f"⚠️ Model file not found at {model_path}")
    except Exception as e:
        print(f"❌ Error loading RF model: {e}")
    
    try:
        # Load vector database for similarity search
        vector_db_path = os.path.join(os.path.dirname(__file__), "vector_db.json")
        if os.path.exists(vector_db_path):
            with open(vector_db_path, "r") as f:
                vector_db = json.load(f)
                print("✅ Vector database loaded successfully")
        else:
            print(f"⚠️ Vector database file not found at {vector_db_path}")
    except Exception as e:
        print(f"❌ Error loading vector database: {e}")
    
    try:
        # Load embedding model for similarity search
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load embedding model: {e}")

# Load models on startup
load_models()

# -----------------------
# API: CALCULATE PREMIUM WITH AI EXPLANATION
# -----------------------
@app.route('/calculate', methods=['POST'])
def calculate():
    """Calculate premium and generate AI explanation"""
    try:
        data = request.get_json()
        
        risk_profile = data.get('risk_profile', {})
        coverage = data.get('coverage', {})
        
        # Predict premium
        premium = predict_premium_with_rf(risk_profile)
        if premium is None:
            # Fallback calculation
            age_factor = risk_profile.get('age', 30) / 100
            smoker_factor = 1.2 if risk_profile.get('smoker') else 1
            sum_insured = coverage.get('sum_insured', 100000)
            premium = round(sum_insured * age_factor * smoker_factor, 2)
            model_used = 'fallback_formula'
        else:
            model_used = 'random_forest'
        
        # Generate AI justification
        justification_data = generate_justification_with_llm(risk_profile, coverage, premium)
        
        # Get similar records
        similar_records = get_similar_records(risk_profile, top_k=3)
        
        return jsonify({
            'success': True,
            'premium': premium,
            'justification': justification_data,
            'similar_records': similar_records,
            'model_used': model_used
        })
    
    except Exception as e:
        print(f"Error in /calculate endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error calculating premium: {str(e)}'
        }), 500

# -------- HISTORY PAGE --------
@app.route('/history', methods=['GET'])
def history():
    if "user_id" not in session:
        return redirect(url_for('login'))
    try:
        cursor.execute(
            "SELECT id, name, age, sex, region, bmi, smoker, sum_insured, premium, justification, created_at FROM calculation_history WHERE user_id=%s ORDER BY created_at DESC",
            (session["user_id"],)
        )
        history_data = cursor.fetchall()
        
        # Parse JSON justification for each record
        for record in history_data:
            try:
                if record.get('justification'):
                    record['justification'] = json.loads(record['justification'])
                else:
                    record['justification'] = []
            except (json.JSONDecodeError, TypeError):
                record['justification'] = []
        
        return render_template('history.html', history=history_data, user_name=session.get("user_name"))
    except Exception as e:
        print(f"Error fetching history: {e}")
        return render_template('history.html', error="Failed to fetch history", user_name=session.get("user_name"))

# -----------------------
# Rule-Based Fallback Justification (No API needed)
# -----------------------
def generate_fallback_justification(risk_profile, coverage, premium):
    """Generate justification using rule-based logic"""
    justifications = []
    
    age = int(risk_profile.get("age", 30))
    bmi = float(risk_profile.get("bmi", 25))
    smoker = risk_profile.get("smoker", False)
    sum_insured = coverage.get("sum_insured", 100000)
    
    # Age factor
    if age < 25:
        age_explanation = "You're under 25, which is typically the lowest risk age group for insurance. This helps keep your premium down."
    elif age < 35:
        age_explanation = "You're in your 20s-30s, a relatively young and low-risk age group for insurance, so you get better rates."
    elif age < 45:
        age_explanation = "You're in your 40s, which is still considered a good age for insurance. Your premium reflects your moderate risk level."
    elif age < 55:
        age_explanation = "You're in your 50s. As people age, insurance risk increases, so premiums tend to be higher."
    elif age < 65:
        age_explanation = "You're in your 60s. The higher premium reflects the increased risk that comes with age."
    else:
        age_explanation = "You're 65 or older. At this age, insurance premiums are significantly higher due to increased health risks."
    
    justifications.append({
        "factor": "Age",
        "explanation": age_explanation
    })
    
    # BMI factor
    if bmi < 18.5:
        bmi_explanation = "Your BMI is below 18.5 (underweight). This is generally considered a positive factor that can help reduce your premium."
    elif bmi < 25:
        bmi_explanation = "Your BMI is 18.5-25 (healthy weight). This is the ideal weight range and helps keep your insurance costs lower."
    elif bmi < 30:
        bmi_explanation = "Your BMI is 25-30 (overweight). This slightly increases your premium compared to a healthy weight range."
    elif bmi < 35:
        bmi_explanation = "Your BMI is 30-35 (obese). This increases your health risk, so your premium is adjusted upward."
    else:
        bmi_explanation = "Your BMI is above 35 (severely obese). This significantly increases your health risks, which is reflected in a higher premium."
    
    justifications.append({
        "factor": "BMI",
        "explanation": bmi_explanation
    })
    
    # Smoking factor
    if smoker:
        smoking_explanation = "You indicated you smoke. Smokers face significantly higher premiums (typically 15-50% more) due to serious health risks associated with smoking."
    else:
        smoking_explanation = "You're a non-smoker. Non-smokers typically get better rates because smoking is a major health risk factor."
    
    justifications.append({
        "factor": "Smoking Status",
        "explanation": smoking_explanation
    })
    
    # Coverage factor
    coverage_explanation = f"Your coverage amount is ${sum_insured:,.0f}. Higher coverage amounts mean higher premiums, as the insurance company takes on more financial risk."
    justifications.append({
        "factor": "Coverage Amount",
        "explanation": coverage_explanation
    })
    
    # Premium summary
    justifications.append({
        "factor": "Final Premium",
        "explanation": f"Your premium of ${premium:,.2f} is calculated by combining all these factors. The main drivers are your age, health metrics (BMI), lifestyle (smoking), and the coverage you've chosen."
    })
    
    return justifications

# -----------------------
# LLM Justification Function (with fallback)
# -----------------------
def generate_justification_with_llm(risk_profile, coverage, premium):
    """Generate AI justification - try OpenAI first, then Gemini, then fallback to rule-based"""
    
    # Try OpenAI first if API key is available
    if OPENAI_API_KEY:
        try:
            print("🤖 Attempting OpenAI API for justification...")
            return generate_openai_justification(risk_profile, coverage, premium)
        except Exception as e:
            print(f"⚠️ OpenAI failed: {type(e).__name__}: {str(e)}")
            print("   Trying Gemini API...")
    else:
        print("📋 No OpenAI API key. Trying Gemini API...")
    
    # Try Gemini if OpenAI failed or not available
    if GEMINI_API_KEY:
        try:
            print("🤖 Attempting Gemini API for justification...")
            return generate_gemini_justification(risk_profile, coverage, premium)
        except Exception as e:
            print(f"⚠️ Gemini failed: {type(e).__name__}: {str(e)}")
            print("   Falling back to rule-based justification")
    else:
        print("📋 No Gemini API key configured.")
    
    # Fallback to rule-based justification
    print("📋 Using rule-based justification generator")
    result = generate_fallback_justification(risk_profile, coverage, premium)
    print(f"✅ Generated {len(result)} justification factors")
    return result

def generate_gemini_justification(risk_profile, coverage, premium):
    """Generate justification using Google Gemini API"""
    prompt_text = f"""
    User Profile:
    - Name: {risk_profile.get('name', 'Unknown')}
    - Age: {risk_profile.get('age', 'Unknown')}
    - Gender: {risk_profile.get('sex', 'Unknown')}
    - BMI: {risk_profile.get('bmi', 'Unknown')}
    - Smoking: {'Yes' if risk_profile.get('smoker') else 'No'}
    - Region: {risk_profile.get('region', 'Unknown')}
    - Coverage: ${coverage.get('sum_insured', 'Unknown')}
    - Premium: ${premium}
    
    Explain in simple, everyday language why this premium amount makes sense. Use words that anyone can understand.
    
    Return ONLY a JSON array with 5 objects. Each must have "factor" and "explanation" fields.
    Return ONLY the JSON array, nothing else.
    """
    
    try:
        GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
        headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
        
        print(f"📤 Trying Gemini 2.0 Flash...")
        response = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=20)
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 429:
            print("⚠️ Rate limited. Waiting and retrying...")
            import time
            time.sleep(2)
            response = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=20)
            print(f"📥 Retry status: {response.status_code}")
        
        if response.status_code != 200:
            response.raise_for_status()
        
        result = response.json()
        print(f"✅ Gemini response received")

        # Extract text from the response
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            content = candidate.get("content", {})
            if isinstance(content, dict) and "parts" in content:
                raw_text = " ".join(part.get("text", "") for part in content["parts"])
            elif isinstance(content, str):
                raw_text = content
            else:
                raw_text = ""
            
            print(f"📝 Response length: {len(raw_text)} chars")
            
            # Parse JSON from response
            try:
                import re
                json_match = re.search(r'\[[\s\S]*\]', raw_text)
                if json_match:
                    json_str = json_match.group(0)
                    justification_data = json.loads(json_str)
                    print(f"✅ Parsed JSON: {len(justification_data)} factors")
                    return justification_data
                else:
                    justification_data = json.loads(raw_text)
                    print(f"✅ Parsed JSON: {len(justification_data)} factors")
                    return justification_data
                    
            except json.JSONDecodeError as je:
                print(f"⚠️ JSON parse error: {str(je)[:50]}")
                raise Exception(f"Failed to parse Gemini response as JSON: {je}")
        else:
            print("⚠️ No candidates in response")
            raise Exception("No candidates in Gemini response")
            
    except Exception as e:
        print(f"❌ Gemini Error: {type(e).__name__}: {str(e)[:70]}")
        raise

def generate_openai_justification(risk_profile, coverage, premium):
    """Generate justification using OpenAI API"""
    from openai import OpenAI
    
    prompt_text = f"""
    User Profile:
    - Name: {risk_profile.get('name', 'Unknown')}
    - Age: {risk_profile.get('age', 'Unknown')}
    - Gender: {risk_profile.get('sex', 'Unknown')}
    - BMI: {risk_profile.get('bmi', 'Unknown')}
    - Smoking: {'Yes' if risk_profile.get('smoker') else 'No'}
    - Region: {risk_profile.get('region', 'Unknown')}
    - Coverage: ${coverage.get('sum_insured', 'Unknown')}
    - Premium: ${premium}
    
    Explain in simple, everyday language why this premium amount makes sense. Use words that anyone can understand.
    
    Return ONLY a JSON array with 5 objects. Each must have "factor" and "explanation" fields:
    [
        {{"factor": "Age", "explanation": "Simple explanation..."}},
        {{"factor": "BMI", "explanation": "Simple explanation..."}},
        {{"factor": "Smoking Status", "explanation": "Simple explanation..."}},
        {{"factor": "Coverage Amount", "explanation": "Simple explanation..."}},
        {{"factor": "Final Premium", "explanation": "Simple explanation..."}}
    ]
    
    Return ONLY the JSON array, nothing else.
    """
    
    try:
        print("🔄 Trying OpenAI API for justification...")
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful insurance advisor. Return valid JSON only."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        raw_text = response.choices[0].message.content.strip()
        print(f"📝 Raw response: {raw_text[:200]}...")
        
        # Parse JSON
        import re
        json_match = re.search(r'\[[\s\S]*\]', raw_text)
        if json_match:
            json_str = json_match.group(0)
            justification_data = json.loads(json_str)
            print(f"✅ OpenAI justification generated successfully")
            return justification_data
        else:
            justification_data = json.loads(raw_text)
            return justification_data
            
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        raise

# -----------------------
# Random Forest Prediction Function
# -----------------------
def predict_premium_with_rf(risk_profile):
    """Predict insurance premium using trained Random Forest model"""
    if not RF_MODEL or not all([LE_SEX, LE_SMOKER, LE_REGION]):
        print("⚠️ Model components missing - RF_MODEL:", bool(RF_MODEL), "LE_SEX:", bool(LE_SEX), "LE_SMOKER:", bool(LE_SMOKER), "LE_REGION:", bool(LE_REGION))
        return None
    
    try:
        age = int(risk_profile.get("age", 30))
        sex = str(risk_profile.get("sex", "male")).lower().strip()
        bmi = float(risk_profile.get("bmi", 25))
        children = int(risk_profile.get("children", 0))
        smoker = risk_profile.get("smoker", False)
        region = str(risk_profile.get("region", "northwest")).lower().strip()
        
        print(f"🔍 Input values: age={age}, sex={sex}, bmi={bmi}, children={children}, smoker={smoker}, region={region}")
        
        # Ensure valid sex value
        if sex not in LE_SEX.classes_:
            print(f"⚠️ Sex '{sex}' not in encoder classes: {LE_SEX.classes_}")
            sex = "male"  # Default to male
        
        # Encode categorical variables
        sex_encoded = LE_SEX.transform([sex])[0]
        smoker_value = "yes" if smoker else "no"
        
        if smoker_value not in LE_SMOKER.classes_:
            print(f"⚠️ Smoker '{smoker_value}' not in encoder classes: {LE_SMOKER.classes_}")
            smoker_value = "no"  # Default to no
            
        smoker_encoded = LE_SMOKER.transform([smoker_value])[0]
        
        # Ensure valid region value
        if region not in LE_REGION.classes_:
            print(f"⚠️ Region '{region}' not in encoder classes: {LE_REGION.classes_}")
            region = "northwest"  # Default
        
        region_encoded = LE_REGION.transform([region])[0]
        
        print(f"✅ Encoded values: sex={sex_encoded}, smoker={smoker_encoded}, region={region_encoded}")
        
        # Prepare input features
        features = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
        print(f"📊 Features array shape: {features.shape}, values: {features}")
        
        # Make prediction
        premium = RF_MODEL.predict(features)[0]
        print(f"✅ Random Forest prediction: ${premium:.2f}")
        return round(premium, 2)
    except Exception as e:
        print(f"❌ Error in Random Forest prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_enhanced_premium(risk_profile, coverage):
    """Calculate insurance premium using comprehensive risk factors"""
    try:
        # Base premium calculation
        sum_insured = coverage.get("sum_insured", 100000)
        base_rate = sum_insured / 100000  # Base rate per $100k
        
        # Age factor
        age = risk_profile.get("age", 30)
        if age < 25:
            age_factor = 0.8
        elif age < 35:
            age_factor = 1.0
        elif age < 45:
            age_factor = 1.2
        elif age < 55:
            age_factor = 1.5
        elif age < 65:
            age_factor = 2.0
        else:
            age_factor = 2.5
        
        # Gender factor
        gender = risk_profile.get("sex", "male").lower()
        gender_factor = 1.1 if gender == "male" else 1.0
        
        # BMI factor
        bmi = risk_profile.get("bmi", 25)
        if bmi < 18.5:
            bmi_factor = 1.1
        elif bmi < 25:
            bmi_factor = 1.0
        elif bmi < 30:
            bmi_factor = 1.3
        elif bmi < 35:
            bmi_factor = 1.6
        else:
            bmi_factor = 2.0
        
        # Smoking factor
        smoker = risk_profile.get("smoker", False)
        smoking_factor = 2.0 if smoker else 1.0
        
        # Medical history factor
        medical_history = risk_profile.get("medical_history", "none")
        if medical_history == "none":
            medical_factor = 1.0
        elif medical_history == "minor":
            medical_factor = 1.2
        elif medical_history == "major":
            medical_factor = 1.5
        else:  # chronic
            medical_factor = 1.8
        
        # Marital status factor
        marital_status = risk_profile.get("marital_status", "single")
        marital_factor = 0.95 if marital_status == "married" else 1.0
        
        # Children factor
        children = risk_profile.get("children", 0)
        children_factor = 1.0 + (children * 0.1) if children > 0 else 0.95
        
        # Job risk factor
        job_risk = risk_profile.get("job_risk", "low")
        if job_risk == "low":
            job_factor = 1.0
        elif job_risk == "medium":
            job_factor = 1.1
        else:  # high
            job_factor = 1.3
        
        # Alcohol consumption factor
        alcohol = risk_profile.get("alcohol_consumption", "none")
        if alcohol == "none":
            alcohol_factor = 1.0
        elif alcohol == "occasional":
            alcohol_factor = 1.05
        elif alcohol == "moderate":
            alcohol_factor = 1.1
        else:  # heavy
            alcohol_factor = 1.3
        
        # Exercise habits factor
        exercise = risk_profile.get("exercise_habits", "moderate")
        if exercise == "active":
            exercise_factor = 0.9
        elif exercise == "moderate":
            exercise_factor = 1.0
        elif exercise == "light":
            exercise_factor = 1.05
        else:  # none
            exercise_factor = 1.1
        
        # Policy term factor
        policy_term = coverage.get("policy_term", 5)
        if policy_term <= 1:
            term_factor = 1.0
        elif policy_term <= 5:
            term_factor = 1.1
        elif policy_term <= 10:
            term_factor = 1.2
        else:
            term_factor = 1.3
        
        # Policy type factor
        policy_type = coverage.get("policy_type", "individual")
        if policy_type == "individual":
            policy_factor = 1.0
        elif policy_type == "family":
            policy_factor = 1.2
        else:  # joint
            policy_factor = 1.1
        
        # Insurance type base adjustment
        insurance_type = risk_profile.get("insurance_type", "health")
        type_base = {
            "health": 1.0,
            "life": 1.2,
            "vehicle": 0.8,
            "home": 0.6,
            "travel": 0.4
        }.get(insurance_type, 1.0)
        
        # Calculate final premium
        premium = (base_rate * age_factor * gender_factor * bmi_factor * smoking_factor * 
                  medical_factor * marital_factor * children_factor * job_factor * 
                  alcohol_factor * exercise_factor * term_factor * policy_factor * type_base)
        
        return round(premium, 2)
        
    except Exception as e:
        print(f"Error in enhanced premium calculation: {e}")
        # Fallback to simple calculation
        sum_insured = coverage.get("sum_insured", 100000)
        age = risk_profile.get("age", 30)
        smoker = risk_profile.get("smoker", False)
        return round(sum_insured * (age / 100) * (1.5 if smoker else 1), 2)

# -----------------------
# PDF Report Generation
# -----------------------
def generate_pdf_report(user_profile, risk_profile, coverage, premium, justification, risk_score, risk_level):
    """Generate a professional PDF report for the insurance quote"""
    try:
        # Create PDF document in memory
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                               rightMargin=0.5*inch, leftMargin=0.5*inch,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        # Container for PDF elements
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6
        )
        
        # Add header
        elements.append(Paragraph("💼 Insurance Premium Quote", title_style))
        elements.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # User Information Section
        elements.append(Paragraph("📋 Personal Information", heading_style))
        user_data = [
            ['Name', user_profile.get('name', 'N/A')],
            ['Email', user_profile.get('email', 'N/A')],
            ['Phone', user_profile.get('phone', 'N/A')],
            ['Age', str(risk_profile.get('age', 'N/A'))],
            ['Gender', risk_profile.get('sex', 'N/A').capitalize()],
            ['Region', risk_profile.get('region', 'N/A').capitalize()],
        ]
        
        user_table = Table(user_data, colWidths=[2*inch, 3.5*inch])
        user_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(user_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Premium Section (Highlighted)
        elements.append(Paragraph("💰 Your Insurance Premium", heading_style))
        premium_style = ParagraphStyle(
            'PremiumStyle',
            parent=styles['Normal'],
            fontSize=32,
            textColor=colors.HexColor('#00d4ff'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        elements.append(Paragraph(f"${premium:,.2f}", premium_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Coverage Details
        elements.append(Paragraph("📊 Coverage Details", heading_style))
        coverage_data = [
            ['Coverage Amount', f"${coverage.get('sum_insured', 0):,.2f}"],
            ['Policy Term', f"{coverage.get('policy_term', 5)} Years"],
            ['Policy Type', coverage.get('policy_type', 'Individual').capitalize()],
            ['Health Metrics', f"BMI: {risk_profile.get('bmi', 'N/A')} | Smoking: {'Yes' if risk_profile.get('smoker') else 'No'}"],
        ]
        
        coverage_table = Table(coverage_data, colWidths=[2*inch, 3.5*inch])
        coverage_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(coverage_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Risk Assessment
        if risk_score:
            elements.append(Paragraph("🏥 Health Risk Assessment", heading_style))
            risk_data = [
                ['Risk Score', str(risk_score)],
                ['Risk Level', risk_level.get('level', 'N/A') if isinstance(risk_level, dict) else 'N/A'],
            ]
            
            risk_table = Table(risk_data, colWidths=[2*inch, 3.5*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ffe0e0')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            elements.append(risk_table)
            elements.append(Spacer(1, 0.3*inch))
        
        # Justification Section
        elements.append(Paragraph("📝 Why This Premium?", heading_style))
        if justification and len(justification) > 0:
            justification_data = [['Factor', 'Explanation']]
            for item in justification:
                factor = item.get('factor', 'Unknown')
                explanation = item.get('explanation', 'N/A')
                # Truncate long explanations for PDF readability
                if len(explanation) > 100:
                    explanation = explanation[:97] + '...'
                justification_data.append([factor, explanation])
            
            justification_table = Table(justification_data, colWidths=[1.5*inch, 4*inch])
            justification_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
            ]))
            elements.append(justification_table)
            elements.append(Spacer(1, 0.2*inch))
        
        # Footer
        elements.append(Spacer(1, 0.3*inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        elements.append(Paragraph(
            "This is a computer-generated insurance quote. Please review all terms and conditions carefully. "
            "For questions, contact our support team.",
            footer_style
        ))
        
        # Build PDF
        doc.build(elements)
        pdf_buffer.seek(0)
        return pdf_buffer
    
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        raise

# -----------------------
# ChromaDB Similar Records Function (JSON-based Vector Search)
# -----------------------
def get_similar_records(risk_profile, top_k=3):
    """Get similar insurance records using vector similarity search or fallback to manual similarity"""
    if not vector_db:
        return []
    
    try:
        # Fallback similarity calculation (doesn't require embedding_model)
        if not embedding_model:
            print("Using fallback similarity search (no embedding model)")
            return get_similar_records_fallback(risk_profile, top_k)
        
        # Primary method: use embedding model
        sex_val = "male" if risk_profile.get("sex", "").lower() != "female" else "female"
        smoker_val = "yes" if risk_profile.get("smoker", False) else "no"
        
        # Create query text
        query_text = f"Age {risk_profile.get('age', 'unknown')}, {sex_val}, BMI {risk_profile.get('bmi', 'unknown')}, smoker {smoker_val}"
        
        # Generate query embedding
        query_embedding = embedding_model.encode(query_text)
        
        # Calculate similarity with all records
        db_embeddings = np.array(vector_db["embeddings"])
        similarities = cosine_similarity([query_embedding], db_embeddings)[0]
        
        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        similar_records = []
        for idx in top_indices:
            record = vector_db["records"][int(idx)]
            similarity_score = float(similarities[int(idx)])
            record["similarity_score"] = round(similarity_score, 4)
            similar_records.append(record)
        
        return similar_records
    except Exception as e:
        print(f"Error in similarity search: {e}")
        # Fallback to manual similarity if embedding fails
        return get_similar_records_fallback(risk_profile, top_k)

def get_similar_records_fallback(risk_profile, top_k=3):
    """Fallback similarity search using manual attribute comparison"""
    if not vector_db or not vector_db.get("records"):
        return []
    
    try:
        query_age = risk_profile.get("age", 30)
        query_bmi = risk_profile.get("bmi", 25)
        query_smoker = risk_profile.get("smoker", False)
        query_sex = risk_profile.get("sex", "male").lower()
        query_region = risk_profile.get("region", "northeast").lower()
        
        # Calculate similarity scores for each record
        scores = []
        for idx, record in enumerate(vector_db["records"]):
            try:
                # Normalize attributes
                age_diff = abs(record.get("age", 30) - query_age)
                bmi_diff = abs(record.get("bmi", 25) - query_bmi)
                
                # Calculate similarity based on differences
                age_similarity = max(0, 1 - (age_diff / 50))  # 50 year range
                bmi_similarity = max(0, 1 - (bmi_diff / 25))  # 25 BMI range
                
                # Exact matches get boost
                smoker_match = 1.0 if (record.get("smoker", False) == query_smoker) else 0.3
                sex_match = 1.0 if (record.get("sex", "").lower() == query_sex) else 0.5
                region_match = 1.0 if (record.get("region", "").lower() == query_region) else 0.7
                
                # Combined similarity (weighted average)
                combined_similarity = (
                    age_similarity * 0.25 +
                    bmi_similarity * 0.20 +
                    smoker_match * 0.25 +
                    sex_match * 0.15 +
                    region_match * 0.15
                )
                
                scores.append((combined_similarity, idx, record))
            except Exception as e:
                print(f"Error calculating similarity for record {idx}: {e}")
                continue
        
        # Sort by similarity and get top K
        scores.sort(key=lambda x: x[0], reverse=True)
        similar_records = []
        for similarity_score, idx, record in scores[:top_k]:
            record_copy = dict(record)
            record_copy["similarity_score"] = round(similarity_score, 4)
            similar_records.append(record_copy)
        
        return similar_records
    except Exception as e:
        print(f"Error in fallback similarity search: {e}")
        return []

# -----------------------
# Novel Features: Savings Tips, Health Score, Future Projections
# -----------------------

def generate_savings_recommendations(risk_profile, current_premium):
    """Generate savings recommendations using LLM or fallback rule-based logic"""
    
    # Try OpenAI if available
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            prompt_text = f"""
            User Profile:
            - Age: {risk_profile.get('age', 'Unknown')}
            - Smoking: {'Yes' if risk_profile.get('smoker') else 'No'}
            - BMI: {risk_profile.get('bmi', 'Unknown')}
            - Region: {risk_profile.get('region', 'Unknown')}
            - Current Premium: ${current_premium}
            
            Provide 3-4 practical recommendations to reduce their insurance premium. Be specific and realistic.
            Format as a simple numbered list.
            """
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            raw_text = response.choices[0].message.content.strip()
            lines = raw_text.split('\n')
            tips = []
            for line in lines:
                line = line.strip()
                if line and any(char.isdigit() for char in line[:3]):
                    cleaned = line.lstrip('0123456789.-) ').strip()
                    if cleaned and len(cleaned) > 10:
                        tips.append(cleaned)
            return tips[:4] if tips else generate_fallback_savings_tips(risk_profile)
        except Exception as e:
            print(f"⚠️ OpenAI savings tips failed: {e}")
    
    # Fallback to rule-based recommendations
    return generate_fallback_savings_tips(risk_profile)

def generate_fallback_savings_tips(risk_profile):
    """Generate savings tips using rule-based logic"""
    tips = []
    
    age = int(risk_profile.get("age", 30))
    bmi = float(risk_profile.get("bmi", 25))
    smoker = risk_profile.get("smoker", False)
    
    # Smoking recommendation (highest impact)
    if smoker:
        tips.append("Quit smoking - This is the single most effective way to reduce your premium by 15-50%")
    else:
        tips.append("Stay a non-smoker - Keep maintaining your non-smoker status for the best rates")
    
    # BMI recommendation
    if bmi >= 30:
        target_bmi = bmi - 5
        tips.append(f"Reduce BMI - Bringing your BMI down to {target_bmi:.0f} or lower can reduce your premium by 10-20%")
    elif bmi >= 25:
        tips.append("Maintain healthy weight - Keep your BMI between 18.5-25 to get the best health rates")
    else:
        tips.append("Keep your healthy weight - You're already in the ideal weight range")
    
    # Exercise recommendation
    tips.append("Exercise regularly - Getting 150 minutes of moderate exercise per week can lower premiums")
    
    # Age-related recommendation
    if age < 45:
        tips.append("Lock in rates early - Your young age now means lower premiums, consider longer-term policies")
    else:
        tips.append("Review coverage regularly - As you age, review your policy to ensure you have adequate coverage")
    
    return tips[:4]

def calculate_health_risk_score(risk_profile):
    """Calculate a health risk score (0-100) based on user profile"""
    score = 50  # Base score
    
    try:
        age = int(risk_profile.get("age", 30))
        bmi = float(risk_profile.get("bmi", 25))
        smoker = risk_profile.get("smoker", False)
        
        # Age factor (younger is better)
        if age < 25:
            score -= 15
        elif age < 40:
            score -= 5
        elif age > 60:
            score += 20
        
        # BMI factor
        if 18.5 <= bmi < 25:
            score -= 15  # Healthy weight
        elif 25 <= bmi < 30:
            score -= 5   # Overweight
        elif bmi >= 30:
            score += 20  # Obese
        
        # Smoking factor
        if smoker:
            score += 25
        
        # Clamp between 0-100
        score = max(0, min(100, score))
    except Exception as e:
        print(f"Error calculating risk score: {e}")
    
    return score

def get_risk_level(score):
    """Convert risk score to risk level"""
    if score < 30:
        return {"level": "Low", "color": "green", "icon": "✅"}
    elif score < 50:
        return {"level": "Moderate", "color": "yellow", "icon": "⚠️"}
    elif score < 75:
        return {"level": "High", "color": "orange", "icon": "⚠️⚠️"}
    else:
        return {"level": "Very High", "color": "red", "icon": "🔴"}

def predict_future_premiums(risk_profile, coverage):
    """Predict premiums at different ages (5, 10, 15 years from now)"""
    current_age = int(risk_profile.get("age", 30))
    future_premiums = {}
    
    try:
        for years_ahead in [5, 10, 15]:
            future_age = current_age + years_ahead
            future_profile = risk_profile.copy()
            future_profile["age"] = future_age
            
            premium = predict_premium_with_rf(future_profile)
            if premium is None:
                # Fallback calculation
                age_factor = future_age / 100
                smoker_factor = 1.2 if risk_profile.get("smoker") else 1
                sum_insured = coverage.get("sum_insured", 100000)
                premium = round(sum_insured * age_factor * smoker_factor, 2)
            
            future_premiums[f"age_{future_age}"] = {
                "age": future_age,
                "premium": round(premium, 2),
                "years_from_now": years_ahead
            }
    except Exception as e:
        print(f"Error predicting future premiums: {e}")
    
    return future_premiums

# -----------------------
# API Routes (HTML Frontend)
# -----------------------

# -------- HOME / INDEX --------
@app.route('/')
def index():
    return render_template('index.html')

# -------- SIGNUP --------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            name = request.form.get("name")
            email = request.form.get("email")
            phone = request.form.get("phone")
            password = request.form.get("password")

            if not name or not email or not password:
                # Check if this is an AJAX request
                if request.headers.get('Content-Type') == 'application/x-www-form-urlencoded' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': False, 'error': 'Name, email and password are required'})
                return render_template('signup.html', error="All fields are required")

            hashed_password = generate_password_hash(password)
            cursor.execute(
                "INSERT INTO users (name,email,password,phone) VALUES (%s,%s,%s,%s)", 
                (name,email,hashed_password,phone)
            )
            db.commit()
            
            # Check if this is an AJAX request
            if request.headers.get('Content-Type') == 'application/x-www-form-urlencoded' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': True, 'message': 'Registration successful! Please login.'})
            return render_template('signup.html', success="Registration successful! Please login.")

        except mysql.connector.IntegrityError:
            if request.headers.get('Content-Type') == 'application/x-www-form-urlencoded' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'error': 'Email already exists'})
            return render_template('signup.html', error="Email already exists")
        except Exception as e:
            print("Signup error:", e)
            if request.headers.get('Content-Type') == 'application/x-www-form-urlencoded' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'error': 'Server error'})
            return render_template('signup.html', error="Server error")
    
    return render_template('signup.html')

# -------- LOGIN --------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email = request.form.get("email")
            password = request.form.get("password")

            if not email or not password:
                # Check if this is an AJAX request
                if request.headers.get('Content-Type') == 'application/x-www-form-urlencoded' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': False, 'error': 'Email and password required'})
                return render_template('login.html', error="Email and password required")

            cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
            user = cursor.fetchone()

            if user and check_password_hash(user["password"], password):
                session["user_id"] = user["id"]
                session["user_name"] = user["name"]
                # Check if this is an AJAX request
                if request.headers.get('Content-Type') == 'application/x-www-form-urlencoded' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': True, 'redirect': url_for('dashboard')})
                return redirect(url_for('dashboard'))
            else:
                # Check if this is an AJAX request
                if request.headers.get('Content-Type') == 'application/x-www-form-urlencoded' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': False, 'error': 'Invalid email or password'})
                return render_template('login.html', error="Invalid email or password")

        except Exception as e:
            print("Login error:", e)
            if request.headers.get('Content-Type') == 'application/x-www-form-urlencoded' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'error': 'Server error'})
            return render_template('login.html', error="Server error")
    
    return render_template('login.html')

# -------- DASHBOARD --------
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    print("DEBUG: Dashboard route called", flush=True)
    if "user_id" not in session:
        return redirect(url_for('login'))
    
    # Get dashboard stats
    total_calculations = 0
    avg_premium = 0
    user_joined = "N/A"
    user_profile = {}
    similar_records = []
    
    try:
        # Fetch user creation date and profile info
        cursor.execute("SELECT name, email, phone, dob, address, city, state, created_at FROM users WHERE id=%s", (session["user_id"],))
        user_data = cursor.fetchone()
        if user_data:
            user_profile = {
                'name': user_data.get('name', ''),
                'email': user_data.get('email', ''),
                'phone': user_data.get('phone', ''),
                'dob': user_data.get('dob', ''),
                'address': user_data.get('address', ''),
                'city': user_data.get('city', ''),
                'state': user_data.get('state', '')
            }
            created_date = user_data.get('created_at')
            if created_date:
                if isinstance(created_date, str):
                    created_date = datetime.strptime(created_date, "%Y-%m-%d %H:%M:%S")
                user_joined = created_date.strftime("%B %d, %Y")
        
        # Fetch calculation stats
        cursor.execute("SELECT COUNT(*) as count, AVG(premium) as avg_prem FROM calculation_history WHERE user_id=%s", (session["user_id"],))
        stats = cursor.fetchone()
        if stats:
            total_calculations = stats.get('count', 0) or 0
            avg_premium = stats.get('avg_prem', 0) or 0
    except Exception as e:
        print(f"Error fetching dashboard stats: {e}")
    
    if request.method == 'POST':
        # Check if this is a similarity search request
        if request.form.get("search_similar") == "true":
            try:
                # Get form data for similarity search
                age = int(request.form.get("sim_age", 30))
                sex = request.form.get("sim_sex", "male")
                region = request.form.get("sim_region", "northeast")
                bmi = float(request.form.get("sim_bmi", 25))
                smoker = request.form.get("sim_smoker") == "yes"
                
                risk_profile = {
                    "age": age,
                    "sex": sex,
                    "region": region,
                    "bmi": bmi,
                    "smoker": smoker,
                }
                
                # Get similar records from vector database
                similar_records = get_similar_records(risk_profile, top_k=6)
                
            except Exception as e:
                print(f"Error in similarity search: {e}")
            
            return render_template('dashboard.html', 
                                    similar_records=similar_records,
                                    user_name=session.get("user_name"),
                                    user_profile=user_profile,
                                    total_calculations=total_calculations,
                                    avg_premium=avg_premium,
                                    user_joined=user_joined)
        
        # Otherwise, process the premium calculation
        try:
            # Get form data
            name = request.form.get("name")
            age = int(request.form.get("age", 30))
            sex = request.form.get("sex", "male")
            region = request.form.get("region", "northwest")
            bmi = float(request.form.get("bmi", 25))
            smoker = request.form.get("smoker") == "on"
            sum_insured = float(request.form.get("sum_insured", 100000))
            
            # New fields
            insurance_type = request.form.get("insuranceType", "health")
            marital_status = request.form.get("maritalStatus", "single")
            location = request.form.get("location", region)  # fallback to region
            policy_term = int(request.form.get("policyTerm", 5))
            policy_type = request.form.get("policyType", "individual")
            children = int(request.form.get("children", 0))
            
            # Health details
            medical_history = request.form.get("medicalHistory", "none")
            existing_diseases = request.form.get("existingDiseases", "None")
            height_str = request.form.get("height")
            height = float(height_str) if height_str else None
            weight_str = request.form.get("weight")
            weight = float(weight_str) if weight_str else None
            
            # Occupation details
            job_type = request.form.get("jobType", "office")
            job_risk = request.form.get("jobRisk", "low")
            
            # Lifestyle details
            alcohol_consumption = request.form.get("alcoholConsumption", "none")
            exercise_habits = request.form.get("exerciseHabits", "moderate")
            
            risk_profile = {
                "name": name,
                "age": age,
                "sex": sex,
                "region": region,
                "bmi": bmi,
                "smoker": smoker,
                "children": children,
                # New fields for enhanced calculation
                "insurance_type": insurance_type,
                "marital_status": marital_status,
                "location": location,
                "policy_term": policy_term,
                "policy_type": policy_type,
                "medical_history": medical_history,
                "existing_diseases": existing_diseases,
                "height": height,
                "weight": weight,
                "job_type": job_type,
                "job_risk": job_risk,
                "alcohol_consumption": alcohol_consumption,
                "exercise_habits": exercise_habits
            }
            
            coverage = {"sum_insured": sum_insured, "policy_term": policy_term, "policy_type": policy_type}
            
            # Predict premium using enhanced calculation with all factors
            premium = calculate_enhanced_premium(risk_profile, coverage)
            model_used = "enhanced_calculation"

            # Generate justification from LLM
            justification_data = generate_justification_with_llm(risk_profile, coverage, premium)
            # Convert to JSON string for database storage
            justification_json = json.dumps(justification_data) if justification_data else "[]"

            # Save calculation history to database
            try:
                cursor.execute(
                    "INSERT INTO calculation_history (user_id, name, age, sex, region, bmi, smoker, sum_insured, premium, justification, insurance_type, marital_status, location, policy_term, policy_type, children, medical_history, existing_diseases, height, weight, job_type, job_risk, alcohol_consumption, exercise_habits) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        session["user_id"],
                        name,
                        age,
                        sex,
                        region,
                        bmi,
                        smoker,
                        sum_insured,
                        premium,
                        justification_json,
                        insurance_type,
                        marital_status,
                        location,
                        policy_term,
                        policy_type,
                        children,
                        medical_history,
                        existing_diseases,
                        height,
                        weight,
                        job_type,
                        job_risk,
                        alcohol_consumption,
                        exercise_habits
                    )
                )
                db.commit()
                print(f"✅ Successfully saved calculation history for user {session['user_id']}")
            except Exception as e:
                print(f"❌ Error saving calculation history: {e}")
                db.rollback()

            # Generate new features for dashboard
            risk_score = calculate_health_risk_score(risk_profile)
            risk_level = get_risk_level(risk_score)
            savings_tips = generate_savings_recommendations(risk_profile, premium)
            future_premiums = predict_future_premiums(risk_profile, coverage)

            return render_template('dashboard.html', 
                                    premium=premium, 
                                    justification=justification_data,
                                    risk_score=risk_score,
                                    risk_level=risk_level,
                                    savings_tips=savings_tips,
                                    future_premiums=future_premiums,
                                    user_name=session.get("user_name"),
                                    user_profile=user_profile,
                                    total_calculations=total_calculations,
                                    avg_premium=avg_premium,
                                    user_joined=user_joined,
                                    similar_records=similar_records)
        
        except Exception as e:
            print(f"Error in calculation: {e}")
            return render_template('dashboard.html', error="Calculation error", user_name=session.get("user_name"),
                                    user_profile=user_profile,
                                    total_calculations=total_calculations,
                                    avg_premium=avg_premium,
                                    user_joined=user_joined,
                                    similar_records=similar_records)
    
    return render_template('dashboard.html', user_name=session.get("user_name"),
                            user_profile=user_profile,
                            total_calculations=total_calculations,
                            avg_premium=avg_premium,
                            user_joined=user_joined,
                            similar_records=similar_records)

# -------- SIMILARITY PAGE --------
@app.route('/similarity', methods=['GET', 'POST'])
def similarity():
    if "user_id" not in session:
        return redirect(url_for('login'))
    
    similar_records = []
    error = None
    
    if request.method == 'POST':
        try:
            # Get form data
            age = int(request.form.get("age", 30))
            sex = request.form.get("sex", "male")
            region = request.form.get("region", "northwest")
            bmi = float(request.form.get("bmi", 25))
            smoker = request.form.get("smoker") == "on"
            
            risk_profile = {
                "age": age,
                "sex": sex,
                "region": region,
                "bmi": bmi,
                "smoker": smoker,
            }
            
            # Get similar records from vector database
            similar_records = get_similar_records(risk_profile, top_k=6)
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            error = "Error searching for similar profiles"
    
    return render_template('similarity.html', similar_records=similar_records, error=error, user_name=session.get("user_name"))


# -------- DOWNLOAD PDF REPORT --------
@app.route('/download_pdf/<int:record_id>', methods=['GET'])
def download_pdf(record_id):
    """Download a PDF report of a specific calculation"""
    if "user_id" not in session:
        return redirect(url_for('login'))
    
    try:
        # Get the calculation record from database
        cursor.execute(
            "SELECT id, name, age, sex, region, bmi, smoker, sum_insured, premium, justification, created_at FROM calculation_history WHERE id=%s AND user_id=%s",
            (record_id, session["user_id"])
        )
        record = cursor.fetchone()
        
        if not record:
            return "Record not found", 404
        
        # Get user profile
        cursor.execute("SELECT name, email, phone FROM users WHERE id=%s", (session["user_id"],))
        user_data = cursor.fetchone()
        
        # Prepare data for PDF
        user_profile = {
            'name': user_data.get('name', 'N/A'),
            'email': user_data.get('email', 'N/A'),
            'phone': user_data.get('phone', 'N/A'),
        }
        
        risk_profile = {
            'name': record.get('name', 'N/A'),
            'age': record.get('age', 30),
            'sex': record.get('sex', 'male'),
            'region': record.get('region', 'northeast'),
            'bmi': record.get('bmi', 25),
            'smoker': record.get('smoker', False),
        }
        
        coverage = {
            'sum_insured': record.get('sum_insured', 100000),
            'policy_term': 5,
            'policy_type': 'Individual',
        }
        
        premium = record.get('premium', 0)
        
        # Parse justification
        try:
            justification = json.loads(record.get('justification', '[]'))
        except:
            justification = []
        
        # Calculate risk score (simple estimation from data)
        risk_score = int((record.get('age', 30) / 100) * (record.get('bmi', 25) / 25) * 50)
        
        risk_level = {
            'level': 'Low' if risk_score < 30 else 'Medium' if risk_score < 60 else 'High',
            'color': 'low' if risk_score < 30 else 'medium' if risk_score < 60 else 'high',
            'icon': '✅' if risk_score < 30 else '⚠️' if risk_score < 60 else '❌'
        }
        
        # Generate PDF
        pdf_buffer = generate_pdf_report(user_profile, risk_profile, coverage, premium, justification, risk_score, risk_level)
        
        # Return PDF file
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"insurance_quote_{record_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
        )
    
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return f"Error generating PDF: {str(e)}", 500


# -------- DELETE HISTORY --------
@app.route('/delete_history/<int:record_id>', methods=['GET', 'POST'])
def delete_history_route(record_id):
    if "user_id" not in session:
        return redirect(url_for('login'))

    try:
        # Verify the record belongs to the current user
        cursor.execute(
            "SELECT user_id FROM calculation_history WHERE id=%s",
            (record_id,)
        )
        record = cursor.fetchone()
        
        if record and record["user_id"] == session["user_id"]:
            # Delete the record
            cursor.execute(
                "DELETE FROM calculation_history WHERE id=%s",
                (record_id,)
            )
            db.commit()
        
        return redirect(url_for('history'))
    except Exception as e:
        print(f"Error deleting history: {e}")
        return redirect(url_for('history'))

# -------- CHARTS PAGE --------
@app.route('/charts', methods=['GET'])
def charts():
    if "user_id" not in session:
        return redirect(url_for('login'))
    
    try:
        cursor.execute(
            "SELECT id, name, age, sex, region, bmi, smoker, sum_insured, premium, created_at FROM calculation_history WHERE user_id=%s ORDER BY created_at DESC",
            (session["user_id"],)
        )
        history_data = cursor.fetchall()
        
        # Calculate stats for charts
        total_calculations = len(history_data)
        avg_premium = sum([h['premium'] for h in history_data]) / total_calculations if total_calculations > 0 else 0
        
        # Get premium data for chart
        premiums = [h['premium'] for h in history_data[:10]]  # Last 10 calculations
        dates = [h['created_at'].strftime('%m-%d') if h['created_at'] else 'N/A' for h in history_data[:10]]
        # Create labels with name and date
        labels = [f"{h['name']} ({h['created_at'].strftime('%m-%d')})" if h['created_at'] else h['name'] for h in history_data[:10]]
        
        # Calculate region-wise statistics
        region_stats = {}
        for h in history_data:
            region = h['region']
            if region not in region_stats:
                region_stats[region] = {'count': 0, 'total_premium': 0, 'avg_premium': 0}
            region_stats[region]['count'] += 1
            region_stats[region]['total_premium'] += h['premium']
        
        # Calculate average premium per region
        for region in region_stats:
            region_stats[region]['avg_premium'] = region_stats[region]['total_premium'] / region_stats[region]['count']
        
        region_names = list(region_stats.keys())
        region_premiums = [region_stats[r]['avg_premium'] for r in region_names]
        
        # Smoker vs Non-smoker statistics
        smokers = [h for h in history_data if h['smoker']]
        non_smokers = [h for h in history_data if not h['smoker']]
        
        smoker_avg_premium = sum([h['premium'] for h in smokers]) / len(smokers) if smokers else 0
        non_smoker_avg_premium = sum([h['premium'] for h in non_smokers]) / len(non_smokers) if non_smokers else 0
        
        # Age group analytics
        age_groups = {
            '18-30': [],
            '31-45': [],
            '46-60': [],
            '61+': []
        }
        
        for h in history_data:
            age = h['age']
            if age <= 30:
                age_groups['18-30'].append(h['premium'])
            elif age <= 45:
                age_groups['31-45'].append(h['premium'])
            elif age <= 60:
                age_groups['46-60'].append(h['premium'])
            else:
                age_groups['61+'].append(h['premium'])
        
        age_group_names = list(age_groups.keys())
        age_group_premiums = [sum(age_groups[g]) / len(age_groups[g]) if age_groups[g] else 0 for g in age_group_names]
        
        return render_template('charts.html', 
                             history=history_data, 
                             user_name=session.get("user_name"),
                             total_calculations=total_calculations,
                             avg_premium=avg_premium,
                             premiums=premiums,
                             dates=dates,
                             labels=labels,
                             region_names=region_names,
                             region_premiums=region_premiums,
                             smoker_count=len(smokers),
                             non_smoker_count=len(non_smokers),
                             smoker_avg=smoker_avg_premium,
                             non_smoker_avg=non_smoker_avg_premium,
                             age_group_names=age_group_names,
                             age_group_premiums=age_group_premiums)
    except Exception as e:
        print(f"Error fetching charts data: {e}")
        return render_template('charts.html', error="Failed to fetch data", user_name=session.get("user_name"))

# -------- EDIT HISTORY --------
@app.route('/edit_history/<int:record_id>', methods=['GET', 'POST'])
def edit_history(record_id):
    if "user_id" not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            age = int(request.form.get('age'))
            sex = request.form.get('sex')
            region = request.form.get('region')
            bmi = float(request.form.get('bmi'))
            smoker = request.form.get('smoker') == 'on'
            sum_insured = float(request.form.get('sum_insured'))
            
            # Recalculate premium with updated data
            risk_profile = {
                "name": name,
                "age": age,
                "sex": sex,
                "region": region,
                "bmi": bmi,
                "smoker": smoker,
                "children": 0
            }
            coverage = {"sum_insured": sum_insured}
            
            # Use Random Forest model for prediction
            new_premium = predict_premium_with_rf(risk_profile)
            if new_premium is None:
                age_factor = age / 100
                smoker_factor = 1.2 if smoker else 1
                new_premium = round(sum_insured * age_factor * smoker_factor, 2)
            else:
                new_premium = round(new_premium, 2)
            
            # Generate new justification
            new_justification_data = generate_justification_with_llm(risk_profile, coverage, new_premium)
            new_justification_json = json.dumps(new_justification_data) if new_justification_data else "[]"
            
            # Update database
            cursor.execute("""
                UPDATE calculation_history 
                SET name=%s, age=%s, sex=%s, region=%s, bmi=%s, smoker=%s, sum_insured=%s, premium=%s, justification=%s
                WHERE id=%s AND user_id=%s
            """, (name, age, sex, region, bmi, smoker, sum_insured, new_premium, new_justification_json, record_id, session["user_id"]))
            
            db.commit()
            return redirect(url_for('history'))
        except Exception as e:
            print(f"Error updating record: {e}")
            return redirect(url_for('history'))
    
    else:
        try:
            cursor.execute("SELECT * FROM calculation_history WHERE id=%s AND user_id=%s", (record_id, session["user_id"]))
            record = cursor.fetchone()
            if not record:
                return redirect(url_for('history'))
            
            # Parse JSON justification
            try:
                if record.get('justification'):
                    record['justification'] = json.loads(record['justification'])
                else:
                    record['justification'] = []
            except json.JSONDecodeError:
                record['justification'] = []
            
            return render_template('edit_history.html', record=record, user_name=session.get("user_name"))
        except Exception as e:
            print(f"Error fetching record: {e}")
            return redirect(url_for('history'))

# -------- API: PREMIUM IMPACT CALCULATOR --------
@app.route('/api/calculate_impact', methods=['POST'])
def calculate_premium_impact():
    """Calculate how changing a single factor affects premium"""
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    try:
        data = request.json
        
        # Base profile
        base_profile = {
            "age": int(data.get("age", 30)),
            "sex": data.get("sex", "male"),
            "bmi": float(data.get("bmi", 25)),
            "region": data.get("region", "northwest"),
            "smoker": data.get("smoker", False),
            "children": int(data.get("children", 0))
        }
        
        sum_insured = float(data.get("sum_insured", 100000))
        
        # Calculate base premium
        base_premium = predict_premium_with_rf(base_profile)
        if base_premium is None:
            age_factor = base_profile["age"] / 100
            smoker_factor = 1.2 if base_profile["smoker"] else 1
            base_premium = round(sum_insured * age_factor * smoker_factor, 2)
        
        # Calculate impact of each factor
        impacts = {}
        
        # Age impact (±5 years)
        age_up = base_profile.copy()
        age_up["age"] = base_profile["age"] + 5
        premium_age_up = predict_premium_with_rf(age_up) or base_premium
        impacts["age"] = round(premium_age_up - base_premium, 2)
        
        # BMI impact (±2 units)
        bmi_up = base_profile.copy()
        bmi_up["bmi"] = base_profile["bmi"] + 2
        premium_bmi_up = predict_premium_with_rf(bmi_up) or base_premium
        impacts["bmi"] = round(premium_bmi_up - base_premium, 2)
        
        # Smoking impact
        smoking_profile = base_profile.copy()
        smoking_profile["smoker"] = not base_profile["smoker"]
        premium_smoking = predict_premium_with_rf(smoking_profile) or base_premium
        impacts["smoking"] = round(premium_smoking - base_premium, 2)
        
        return jsonify({
            "success": True,
            "base_premium": base_premium,
            "impacts": impacts
        })
    
    except Exception as e:
        print(f"Error in premium impact calculation: {e}")
        return jsonify({"error": str(e)}), 500

# -------- SETTINGS API ENDPOINTS --------
@app.route('/api/update-profile', methods=['POST'])
def update_profile():
    """Update user profile information"""
    try:
        if "user_id" not in session:
            return jsonify({"success": False, "message": "Not authorized"}), 401
        
        data = request.get_json()
        user_id = session.get("user_id")
        
        # Update user in database
        update_query = """
        UPDATE users 
        SET name = %s, email = %s, phone = %s, dob = %s, address = %s, city = %s, state = %s
        WHERE id = %s
        """
        
        cursor.execute(update_query, (
            data.get('fullName'),
            data.get('email'),
            data.get('phone'),
            data.get('dob'),
            data.get('address'),
            data.get('city'),
            data.get('state'),
            user_id
        ))
        
        db.commit()
        
        # Update session
        session['user_name'] = data.get('fullName')
        
        return jsonify({"success": True, "message": "Profile updated successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/update-preferences', methods=['POST'])
def update_preferences():
    """Update user preferences"""
    try:
        if "user_id" not in session:
            return jsonify({"success": False, "message": "Not authorized"}), 401
        
        data = request.get_json()
        
        # Store preferences in session for now
        preferences = {
            'theme': data.get('theme'),
            'dateFormat': data.get('dateFormat'),
            'currency': data.get('currency'),
            'language': data.get('language')
        }
        
        session['preferences'] = preferences
        
        return jsonify({"success": True, "message": "Preferences updated successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/update-notifications', methods=['POST'])
def update_notifications():
    """Update notification settings"""
    try:
        if "user_id" not in session:
            return jsonify({"success": False, "message": "Not authorized"}), 401
        
        data = request.get_json()
        
        # Store notification preferences in session
        notifications = {
            'emailNotif': data.get('emailNotif'),
            'renewalReminders': data.get('renewalReminders'),
            'priceAlerts': data.get('priceAlerts'),
            'marketingComms': data.get('marketingComms'),
            'smsNotif': data.get('smsNotif')
        }
        
        session['notifications'] = notifications
        
        return jsonify({"success": True, "message": "Notification settings updated"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/update-password', methods=['POST'])
def update_password():
    """Update user password"""
    try:
        if "user_id" not in session:
            return jsonify({"success": False, "message": "Not authorized"}), 401
        
        data = request.get_json()
        user_id = session.get("user_id")
        current_password = data.get('currentPassword')
        new_password = data.get('newPassword')
        
        # Get current password hash
        cursor.execute("SELECT password FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        
        if not result:
            return jsonify({"success": False, "message": "User not found"}), 404
        
        # Verify current password
        if not check_password_hash(result["password"], current_password):
            return jsonify({"success": False, "message": "Current password is incorrect"}), 401
        
        # Update password
        hashed_password = generate_password_hash(new_password)
        update_query = "UPDATE users SET password = %s WHERE id = %s"
        cursor.execute(update_query, (hashed_password, user_id))
        
        db.commit()
        
        return jsonify({"success": True, "message": "Password updated successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/delete-account', methods=['POST'])
def delete_account():
    """Delete user account permanently"""
    try:
        if "user_id" not in session:
            return jsonify({"success": False, "message": "Not authorized"}), 401
        
        user_id = session.get("user_id")
        
        # Delete calculation history first (foreign key constraint)
        cursor.execute("DELETE FROM calculation_history WHERE user_id = %s", (user_id,))
        
        # Delete user
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        
        db.commit()
        
        # Clear session
        session.clear()
        
        return jsonify({"success": True, "message": "Account deleted"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# -------- QUICK QUOTE PAGE --------
@app.route('/quick_quote')
def quick_quote():
    if "user_id" not in session:
        return redirect(url_for('login'))
    return render_template('quick_quote.html', user_name=session.get("user_name"))

# -------- SETTINGS PAGE --------
@app.route('/settings')
def settings():
    if "user_id" not in session:
        return redirect(url_for('login'))
    
    # Fetch full user profile from database
    user_profile = {}
    try:
        cursor.execute("SELECT name, email, phone, dob, address, city, state FROM users WHERE id=%s", (session["user_id"],))
        user_data = cursor.fetchone()
        if user_data:
            user_profile = {
                'name': user_data.get('name', ''),
                'email': user_data.get('email', ''),
                'phone': user_data.get('phone', ''),
                'dob': user_data.get('dob', ''),
                'address': user_data.get('address', ''),
                'city': user_data.get('city', ''),
                'state': user_data.get('state', '')
            }
    except Exception as e:
        print(f"Error fetching user profile: {e}")
    
    return render_template('settings.html', user_name=session.get("user_name"), user_profile=user_profile)

# -------- WELLNESS HUB PAGE --------
@app.route('/wellness')
def wellness():
    if "user_id" not in session:
        return redirect(url_for('login'))
    
    return render_template('wellness.html', 
                          user_name=session.get("user_name"))

# -------- FAQ PAGE --------
@app.route('/faq')
def faq():
    return render_template('faq.html', user_name=session.get("user_name"))

# -------- LOGOUT --------
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    return redirect(url_for('index'))

# -----------------------
# Run App
# -----------------------
if __name__=="__main__":
    app.run(debug=True, port=5000)
