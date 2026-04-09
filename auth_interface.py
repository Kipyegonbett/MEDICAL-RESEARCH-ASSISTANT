import streamlit as st
import numpy as np
import pandas as pd
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import os
import gdown
from database_manager import DatabaseManager

# ── Constants for your model ──────────────────────────────────────────────────
MODEL_NAME     = 'dmis-lab/biobert-base-cased-v1.2'
CHECKPOINT     = 'biobert_icd11_best.pt'
ENCODER        = 'icd11_label_encoder.pickle'
MAX_LEN        = 256
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Paste your Google Drive File IDs here ─────────────────────────────────────
MODEL_FILE_ID   = '1nWNKgHzwlrOQqWz9GmCCNK6BUkw932jJ'
ENCODER_FILE_ID = '1VvnRyzxoniUzII0Vxxj4FKl1UVvoLM1W'

ICD11_CODES = {
    'Certain infectious or parasitic diseases'                          : '1',
    'Neoplasms'                                                         : '2',
    'Diseases of the blood or blood-forming organs'                     : '3',
    'Diseases of the immune system'                                     : '4',
    'Endocrine, nutritional or metabolic diseases'                      : '5',
    'Mental, behavioural or neurodevelopmental disorders'               : '6',
    'Sleep-wake disorders'                                              : '7',
    'Diseases of the nervous system'                                    : '8',
    'Diseases of the visual system'                                     : '9',
    'Diseases of the ear or mastoid process'                            : '10',
    'Diseases of the circulatory system'                                : '11',
    'Diseases of the respiratory system'                                : '12',
    'Diseases of the digestive system'                                  : '13',
    'Diseases of the skin'                                              : '14',
    'Diseases of the musculoskeletal system or connective tissue'       : '15',
    'Diseases of the genitourinary system'                              : '16',
    'Pregnancy, childbirth or the puerperium'                           : '17',
    'Certain conditions originating in the perinatal period'            : '18',
    'Developmental anomalies'                                           : '19',
    'Symptoms, signs or clinical findings, not elsewhere classified'    : '21',
    'Injury, poisoning or certain other consequences of external causes': '22',
    'Conditions related to sexual health'                               : '23',
}

QUICK_EXAMPLES = {
    'STEMI (Circulatory)':
        '75yo male, sudden onset crushing chest pain radiating to left arm, diaphoretic, HR 110, BP 88/60, ECG ST elevation V2-V5, troponin 18.4, cath lab activated, LAD occlusion stented',
    'Pre-eclampsia (Pregnancy)':
        '28F, 36/40 weeks gestation, severe headache and visual disturbances, BP 170/110, urine 3+ protein, brisk reflexes with clonus, platelets 98, MgSO4 loading dose given, emergency LSCS prepared',
    'Depression (Mental health)':
        '34yo male, 6 week history of persistent low mood, unable to get out of bed, lost interest in all activities, early morning wakening, PHQ-9 score 24, passive suicidal ideation, sertraline commenced',
    'Pneumonia (Respiratory)':
        '67yo female, productive cough with green sputum, fever 38.9, right lower lobe consolidation on CXR, WBC 18.4, CRP 280, CURB-65 score 3, IV ceftriaxone commenced, O2 via venturi mask',
    'Neonatal jaundice (Perinatal)':
        'Neonate day 3, jaundice visible to abdomen, SBR 280 umol/L above treatment threshold, breastfeeding encouraged, phototherapy commenced, recheck SBR in 6 hours',
}

# ── Download model files ───────────────────────────────────────────────────────
def download_files():
    if not os.path.exists(CHECKPOINT):
        with st.spinner('Downloading model weights — please wait...'):
            url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}&export=download&confirm=t'
            gdown.download(url, CHECKPOINT, quiet=False, fuzzy=True)

    if not os.path.exists(ENCODER):
        with st.spinner('Downloading label encoder...'):
            url = f'https://drive.google.com/uc?id={ENCODER_FILE_ID}&export=download&confirm=t'
            gdown.download(url, ENCODER, quiet=False, fuzzy=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(ENCODER, 'rb') as f:
        label_encoder = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_)
    )
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    return model, tokenizer, label_encoder

# ── Prediction ─────────────────────────────────────────────────────────────────
def predict_note(text, model, tokenizer, label_encoder):
    if not text.strip():
        return None

    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids      = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    top_idx        = int(np.argmax(probs))
    top_label      = label_encoder.classes_[top_idx]
    top_confidence = float(probs[top_idx]) * 100
    icd_code       = ICD11_CODES.get(top_label, 'N/A')

    return {
        'chapter'    : top_label,
        'icd_code'   : icd_code,
        'confidence' : top_confidence,
    }

def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'show_registration' not in st.session_state:
        st.session_state.show_registration = False

def login_page(db):
    """Display login page"""
    st.title("🏥 Hospital Management System")
    st.subheader("ICD-11 Medical Notes Classifier")
    st.markdown("Please login to access the ICD-11 classification system")
    
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="name@hospital.ac.ke")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login", use_container_width=True)
        
        if submit:
            if email and password:
                user, message = db.login_user(email, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_email = user[1]
                    st.session_state.user_name = user[2]
                    st.session_state.user_role = user[3]
                    st.session_state.user_id = user[0]
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("Please enter both email and password")
    
    # Registration link
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📝 Don't have an account? Register here", use_container_width=True):
            st.session_state.show_registration = True
            st.rerun()

def registration_page(db):
    """Display registration page"""
    st.title("📝 Register New Account")
    st.markdown("**Note:** Email must end with **@hospital.ac.ke**")
    
    with st.form("registration_form"):
        full_name = st.text_input("Full Name")
        email = st.text_input("Email", placeholder="name@hospital.ac.ke")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        submit = st.form_submit_button("Register", use_container_width=True)
        
        if submit:
            if not full_name or not email or not password:
                st.warning("Please fill all fields")
            elif not email.endswith('@hospital.ac.ke'):
                st.error("Email must end with @hospital.ac.ke")
            elif password != confirm_password:
                st.error("Passwords do not match")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                success, message = db.register_user(email, password, full_name)
                if success:
                    st.success(message)
                    st.info("You will be notified when admin approves your account.")
                    if st.button("Back to Login"):
                        st.session_state.show_registration = False
                        st.rerun()
                else:
                    st.error(message)
    
    if st.button("← Back to Login"):
        st.session_state.show_registration = False
        st.rerun()

def user_icd11_interface():
    """Your complete ICD-11 classifier interface"""
    # Load the model
    download_files()
    
    with st.spinner('Loading BioBERT model...'):
        try:
            model, tokenizer, label_encoder = load_model()
            model_loaded = True
        except Exception as e:
            model_loaded = False
            model_error = str(e)
    
    # Sidebar with user info
    with st.sidebar:
        st.markdown(f"### 👤 User: {st.session_state.user_name}")
        st.markdown(f"📧 {st.session_state.user_email}")
        st.divider()
    
    # Your existing CSS (keeping all your styles)
    st.markdown("""
    <style>
        /* Hero banner */
        .hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem 2rem;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 32px rgba(102,126,234,0.3);
        }
        .hero h1 {
            color: white;
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .hero p {
            color: rgba(255,255,255,0.85);
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
            color: white;
            box-shadow: 0 4px 16px rgba(102,126,234,0.25);
        }
        .metric-card .label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.85;
        }
        .metric-card .value {
            font-size: 2rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }
        .result-card {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            border-radius: 12px;
            padding: 1.5rem;
            color: white;
            box-shadow: 0 4px 16px rgba(17,153,142,0.3);
            margin-top: 1rem;
        }
        .result-card-warn {
            background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
            border-radius: 12px;
            padding: 1.5rem;
            color: white;
            box-shadow: 0 4px 16px rgba(247,151,30,0.3);
            margin-top: 1rem;
        }
        .section-header {
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .step {
            background: #1e1e2e;
            border-left: 4px solid #667eea;
            border-radius: 0 8px 8px 0;
            padding: 0.7rem 1rem;
            margin-bottom: 0.5rem;
            font-size: 0.95rem;
        }
        .sidebar-card {
            background: #1e1e2e;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #2e2e4e;
        }
        .chapter-item {
            background: #1e1e2e;
            border-radius: 8px;
            padding: 0.5rem 0.8rem;
            margin-bottom: 0.4rem;
            font-size: 0.82rem;
            border-left: 3px solid #667eea;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Hero banner
    st.markdown("""
    <div class="hero">
        <h1>🏥 Medical Notes ICD-11 Classifier</h1>
        <p>AI-Powered Classification into ICD-11 Chapters using BioBERT</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main interface columns
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.markdown('<div class="section-header">📄 Input Medical Notes</div>', unsafe_allow_html=True)
        
        # Quick example selector
        selected_example = st.selectbox(
            'Quick Examples',
            ['Select an example...'] + list(QUICK_EXAMPLES.keys())
        )
        
        # File upload
        uploaded_txt = st.file_uploader('Or upload a TXT file', type=['txt'])
        
        # Determine default text
        default_text = ''
        if selected_example != 'Select an example...':
            default_text = QUICK_EXAMPLES[selected_example]
        elif uploaded_txt is not None:
            default_text = uploaded_txt.read().decode('utf-8')
        
        note_input = st.text_area(
            'Medical Notes',
            value=default_text,
            placeholder='Enter medical notes describing patient symptoms, history, examination findings, and clinical impression...',
            height=220
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            classify_btn = st.button('🔍 Classify Notes', use_container_width=True, type='primary')
        with col2:
            clear_btn = st.button('🗑️ Clear', use_container_width=True)
        
        if clear_btn:
            st.rerun()
        
        # How to use
        st.divider()
        with st.expander('📖 How to Use', expanded=True):
            steps = [
                ('1', 'Upload a text file OR paste medical notes'),
                ('2', 'Or select an example from the dropdown'),
                ('3', 'Click "Classify Notes" to analyse'),
                ('4', 'Review the ICD-11 chapter classification'),
            ]
            for num, text in steps:
                st.markdown(f'<div class="step"><b>{num}.</b> {text}</div>', unsafe_allow_html=True)
    
    with right_col:
        st.markdown('<div class="section-header">📈 Model Metrics</div>', unsafe_allow_html=True)
        
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("""
            <div class="metric-card">
                <div class="label">ICD-11 Chapters</div>
                <div class="value">22</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown("""
            <div class="metric-card">
                <div class="label">Test Accuracy</div>
                <div class="value">100%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<br>', unsafe_allow_html=True)
        
        m3, m4 = st.columns(2)
        with m3:
            st.markdown("""
            <div class="metric-card">
                <div class="label">Training Samples</div>
                <div class="value">11K</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Device</div>
                <div class="value" style="font-size:1rem">{str(DEVICE).upper()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        st.markdown('<div class="section-header">🎯 Classification Result</div>', unsafe_allow_html=True)
        
        if classify_btn:
            if not model_loaded:
                st.error('Model not loaded. Please refresh the page.')
            elif not note_input.strip():
                st.warning('Please enter or upload a clinical note first.')
            else:
                with st.spinner('Analysing with BioBERT...'):
                    result = predict_note(note_input, model, tokenizer, label_encoder)
                    time.sleep(0.4)
                
                conf = result['confidence']
                
                if conf >= 70:
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="chapter">📋 {result['chapter']}</div>
                        <div class="code">ICD-11 Chapter {result['icd_code']}</div>
                        <div class="confidence">Confidence: {conf:.1f}% 🟢</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card-warn">
                        <div class="chapter">📋 {result['chapter']}</div>
                        <div class="code">ICD-11 Chapter {result['icd_code']}</div>
                        <div class="confidence">Confidence: {conf:.1f}% ⚠️</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.progress(conf / 100)
        else:
            st.info('Enter a clinical note and click **Classify Notes** to see the result here.')
    
    # Batch processing section
    st.divider()
    st.markdown('<div class="section-header">📋 Batch Processing</div>', unsafe_allow_html=True)
    
    with st.expander('Process multiple notes at once via CSV upload'):
        st.markdown('Upload a **CSV file** with a column named `text` — one clinical note per row.')
        
        template_df = pd.DataFrame({'text': [
            '75yo male, crushing chest pain, ST elevation V2-V5, troponin 18.4, cath lab activated',
            '28F, 36/40 gestation, BP 170/110, urine 3+ protein, MgSO4 commenced',
            '34yo male, low mood 6 weeks, PHQ-9 score 24, sertraline commenced',
        ]})
        st.download_button(
            label='⬇️ Download CSV Template',
            data=template_df.to_csv(index=False),
            file_name='icd11_template.csv',
            mime='text/csv'
        )
        
        uploaded_csv = st.file_uploader('Upload CSV', type=['csv'], key='batch_csv')
        
        if uploaded_csv:
            try:
                df = pd.read_csv(uploaded_csv)
                if 'text' not in df.columns:
                    st.error('CSV must have a column named "text".')
                else:
                    st.success(f'✓ {len(df):,} notes loaded')
                    st.dataframe(df.head(3), use_container_width=True)
                    
                    if st.button('🚀 Run Batch Classification', type='primary'):
                        results = []
                        progress = st.progress(0)
                        status = st.empty()
                        total = len(df)
                        
                        for i, row in df.iterrows():
                            status.text(f'Processing note {i+1} of {total}...')
                            result = predict_note(str(row['text']), model, tokenizer, label_encoder)
                            if result:
                                results.append({
                                    'note': str(row['text'])[:80] + '...',
                                    'chapter': result['chapter'],
                                    'icd_code': f"Chapter {result['icd_code']}",
                                    'confidence_%': round(result['confidence'], 1),
                                    'flag': '⚠️ Review' if result['confidence'] < 70 else '✅ OK'
                                })
                            else:
                                results.append({
                                    'note': str(row['text'])[:80] + '...',
                                    'chapter': 'ERROR — empty note',
                                    'icd_code': 'N/A',
                                    'confidence_%': 0.0,
                                    'flag': '⚠️ Review'
                                })
                            progress.progress((i + 1) / total)
                        
                        status.text('✓ Done')
                        results_df = pd.DataFrame(results)
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric('Total Notes', total)
                        c2.metric('Avg Confidence', f"{results_df['confidence_%'].mean():.1f}%")
                        c3.metric('Flagged for Review', int((results_df['flag'] == '⚠️ Review').sum()))
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        st.download_button(
                            label='⬇️ Download Results CSV',
                            data=results_df.to_csv(index=False),
                            file_name='icd11_results.csv',
                            mime='text/csv'
                        )
            except Exception as e:
                st.error(f'Error: {e}')
    
    # Footer
    st.divider()
    st.markdown(
        '<center><small>ICD-11 Chapter Classifier — BioBERT fine-tuned — For research and educational use only</small></center>',
        unsafe_allow_html=True
    )
    
    # Logout button
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        for key in ['logged_in', 'user_email', 'user_name', 'user_role', 'user_id']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

def admin_panel(db):
    """Admin panel for user management"""
    st.sidebar.success(f"👑 Admin: {st.session_state.user_name}")
    
    st.title("⚙️ Admin Dashboard")
    st.markdown("Manage user accounts and approvals")
    
    tab1, tab2, tab3 = st.tabs(["📋 Pending Approvals", "👥 Manage Users", "📊 Statistics"])
    
    with tab1:
        st.header("Users Pending Approval")
        pending_users = db.get_pending_users()
        
        if pending_users:
            for user in pending_users:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 3, 2, 1])
                    with col1:
                        st.write(f"**{user[2]}**")
                    with col2:
                        st.write(user[1])
                    with col3:
                        st.write(f"Requested: {user[3][:10]}")
                    with col4:
                        if st.button("✅ Approve", key=f"approve_{user[0]}"):
                            if db.approve_user(user[0]):
                                st.success(f"Approved {user[2]}")
                                st.rerun()
                    st.divider()
        else:
            st.info("No pending approvals")
    
    with tab2:
        st.header("All Users")
        all_users = db.get_all_users()
        
        if all_users:
            for user in all_users:
                with st.expander(f"{user[2]} - {user[1]}"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**Role:** {user[3]}")
                        st.write(f"**Status:** {'✅ Approved' if user[4] else '⏳ Pending'}")
                        st.write(f"**Joined:** {user[5][:10]}")
                    
                    with col2:
                        if user[3] != 'admin':
                            new_role = st.selectbox(
                                "Change Role",
                                ['user', 'admin'],
                                index=0 if user[3] == 'user' else 1,
                                key=f"role_{user[0]}"
                            )
                            if new_role != user[3]:
                                if db.change_user_role(user[0], new_role):
                                    st.success(f"Role changed to {new_role}")
                                    st.rerun()
                    
                    with col3:
                        if st.button("🗑️ Delete", key=f"delete_{user[0]}"):
                            success, message = db.delete_user(user[0])
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
        else:
            st.info("No users found")
    
    with tab3:
        st.header("System Statistics")
        all_users = db.get_all_users()
        pending_users = db.get_pending_users()
        
        total_users = len(all_users)
        approved_users = sum(1 for u in all_users if u[4])
        pending_count = len(pending_users)
        admin_count = sum(1 for u in all_users if u[3] == 'admin')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", total_users)
        with col2:
            st.metric("Approved Users", approved_users)
        with col3:
            st.metric("Pending Approval", pending_count)
        with col4:
            st.metric("Admins", admin_count)
    
    # Logout button
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        for key in ['logged_in', 'user_email', 'user_name', 'user_role', 'user_id']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

def main_auth_interface():
    """Main authentication flow controller"""
    db = DatabaseManager()
    init_session_state()
    
    if not st.session_state.logged_in:
        if not st.session_state.show_registration:
            login_page(db)
        else:
            registration_page(db)
    else:
        if st.session_state.user_role == 'admin':
            admin_panel(db)
        else:
            user_icd11_interface()
