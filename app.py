import os
import secrets

# Generate a secret key if needed
# print(secrets.token_hex(16))

try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using default environment variables.")

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from vsrl.vsrl_core import analyze_educational_image

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'vsrl_secret_key_12345')

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Also create results folder
os.makedirs('static/results', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        try:
            result = analyze_educational_image(filepath)
            
            # Return results page
            return render_template('results.html', 
                                  image_file=f"uploads/{filename}",
                                  visualization_file=f"results/{result['visualization_file']}",
                                  explanation=result['explanation'],
                                  vsrl_data=result['vsrl_annotation'])
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            flash(f'Error analyzing image: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF).')
        return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/test-config')
def test_config():
    if app.debug:  # Only show in debug mode for security
        config_info = {
            'FLASK_APP': os.getenv('FLASK_APP'),
            'FLASK_ENV': os.getenv('FLASK_ENV'),
            'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY')[:5] + '...' if os.getenv('GEMINI_API_KEY') else None,
            'SECRET_KEY': 'Set' if app.secret_key else 'Not set'
        }
        return jsonify(config_info)
    return "Configuration check not available in production mode", 403

if __name__ == '__main__':
    app.run(debug=True)