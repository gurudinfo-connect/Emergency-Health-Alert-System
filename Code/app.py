from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
app.secret_key = 'neuro_secret_key_x99'

# --- Database Config ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
db = SQLAlchemy(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# --- Load Models (Error Handling Added) ---
try:
    heart_model = pickle.load(open('models/heart_model.pkl', 'rb'))
    kidney_model = pickle.load(open('models/kidney_model.pkl', 'rb'))
    skin_model = tf.keras.models.load_model('models/skin_model.h5')
    print("✅ All 3 Models loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Models not found. Ensure 'models/' folder exists. {e}")
    heart_model = kidney_model = skin_model = None

# Skin Classes
SKIN_CLASSES = [
    'Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 
    'Dermatofibroma', 'Melanocytic nevus', 'Melanoma', 
    'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis', 'Vascular lesion'
]

# --- User Model ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='user')

# --- Helper Functions ---
def get_code(value, options):
    options = sorted(options)
    try:
        return options.index(value)
    except ValueError:
        return 0

def check_auth():
    return 'user_id' in session

# --- Routes ---

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/base_paper')
def base_paper():
    return render_template('base_paper.html')

@app.route('/predict')
def predict():
    if not check_auth(): return redirect(url_for('login'))
    return render_template('predict.html') # The Dashboard to choose disease

# --- HEART DISEASE ROUTE ---
@app.route('/predict/heart', methods=['GET', 'POST'])
def predict_heart():
    if not check_auth(): return redirect(url_for('login'))
    
    prediction = None
    alert_msg = None
    
    if request.method == 'POST':
        try:
            # Gather 13 inputs
            features = [
                float(request.form['age']),
                get_code(request.form['sex'], ['Female', 'Male']),
                get_code(request.form['cp'], ['asymptomatic', 'atypical angina', 'non-anginal', 'typical angina']),
                float(request.form['trestbps']),
                float(request.form['chol']),
                get_code(request.form['fbs'], ['FALSE', 'TRUE']),
                get_code(request.form['restecg'], ['lv hypertrophy', 'normal', 'st-t abnormality']),
                float(request.form['thalach']),
                get_code(request.form['exang'], ['FALSE', 'TRUE']),
                float(request.form['oldpeak']),
                get_code(request.form['slope'], ['downsloping', 'flat', 'upsloping']),
                float(request.form['ca']),
                get_code(request.form['thal'], ['fixed defect', 'normal', 'reversable defect'])
            ]
            
            final_features = np.array([features])
            pred = heart_model.predict(final_features)
            
            if pred[0] == 1:
                prediction = "High Risk (Positive)"
                alert_msg = "EMERGENCY: High Heart Disease Risk Detected!"
            else:
                prediction = "Low Risk (Negative)"
                
        except Exception as e:
            flash(f"Error: {e}", 'error')

    return render_template('tool_heart.html', prediction=prediction, alert_msg=alert_msg)

# --- KIDNEY DISEASE ROUTE ---
@app.route('/predict/kidney', methods=['GET', 'POST'])
def predict_kidney():
    if not check_auth(): return redirect(url_for('login'))
    
    prediction = None
    alert_msg = None
    
    if request.method == 'POST':
        try:
            # Gather 24 inputs
            # Numeric
            nums = [float(request.form[f]) for f in ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']]
            # Categorical
            cats = [
                get_code(request.form['rbc'], ['abnormal', 'normal']),
                get_code(request.form['pc'], ['abnormal', 'normal']),
                get_code(request.form['pcc'], ['notpresent', 'present']),
                get_code(request.form['ba'], ['notpresent', 'present']),
                get_code(request.form['htn'], ['no', 'yes']),
                get_code(request.form['dm'], ['no', 'yes']),
                get_code(request.form['cad'], ['no', 'yes']),
                get_code(request.form['appet'], ['good', 'poor']),
                get_code(request.form['pe'], ['no', 'yes']),
                get_code(request.form['ane'], ['no', 'yes'])
            ]
            
            # Combine: nums (0-13) + cats (14-23) -> Correct order depends on training
            # Based on previous Train.py, order was: age,bp,sg,al,su, rbc,pc,pcc,ba, bgr,bu,sc,sod,pot,hemo,pcv,wc,rc, htn,dm,cad,appet,pe,ane
            # Let's reconstruct carefully
            row = [nums[0], nums[1], nums[2], nums[3], nums[4], # age to su
                   cats[0], cats[1], cats[2], cats[3], # rbc to ba
                   nums[5], nums[6], nums[7], nums[8], nums[9], nums[10], nums[11], nums[12], nums[13], # bgr to rc
                   cats[4], cats[5], cats[6], cats[7], cats[8], cats[9]] # htn to ane
            
            final_features = np.array([row])
            pred = kidney_model.predict(final_features)
            
            if pred[0] == 1:
                prediction = "Chronic Kidney Disease Detected"
                alert_msg = "EMERGENCY: High Kidney Disease Risk!"
            else:
                prediction = "Healthy (No Disease Detected)"

        except Exception as e:
            flash(f"Error: {e}", 'error')

    return render_template('tool_kidney.html', prediction=prediction, alert_msg=alert_msg)

# --- SKIN CANCER ROUTE ---
@app.route('/predict/skin', methods=['GET', 'POST'])
def predict_skin():
    if not check_auth(): return redirect(url_for('login'))
    
    prediction = None
    confidence = None
    image_url = None
    alert_msg = None
    
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_url = filepath
            
            try:
                img = image.load_img(filepath, target_size=(128, 128))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                
                preds = skin_model.predict(img_array)
                idx = np.argmax(preds)
                prediction = SKIN_CLASSES[idx]
                confidence = round(np.max(preds) * 100, 2)
                
                dangerous = ['Melanoma', 'Squamous cell carcinoma', 'Basal cell carcinoma']
                if prediction in dangerous:
                    alert_msg = f"ALERT: Malignant Condition Detected ({prediction})"
            except Exception as e:
                flash(f"Error processing image: {e}", 'error')
                
    return render_template('tool_skin.html', prediction=prediction, confidence=confidence, image_url=image_url, alert_msg=alert_msg)

# --- AUTH & ADMIN ROUTES ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username exists!', 'error')
            return redirect(url_for('register'))
        db.session.add(User(username=username, password=generate_password_hash(password, method='pbkdf2:sha256')))
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            session['user_id'] = user.id
            session['role'] = user.role
            return redirect(url_for('admin_dashboard') if user.role == 'admin' else url_for('predict'))
        flash('Invalid Credentials', 'error')
    return render_template('login.html')

@app.route('/admin')
def admin_dashboard():
    if session.get('role') != 'admin': return redirect(url_for('login'))
    return render_template('admin.html', users=User.query.all())

@app.route('/delete_user/<int:id>')
def delete_user(id):
    if session.get('role') != 'admin': return redirect(url_for('login'))
    User.query.filter_by(id=id).delete()
    db.session.commit()
    return redirect(url_for('admin_dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

def init_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            db.session.add(User(username='admin', password=generate_password_hash('admin123', method='pbkdf2:sha256'), role='admin'))
            db.session.commit()

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
