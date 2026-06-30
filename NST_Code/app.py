import os
import uuid
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
import io

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

import threading
import time

def start_cleanup_thread(upload_folder, max_age_seconds=1800, interval_seconds=300):
    def cleanup_loop():
        # Avoid running double threads in Werkzeug reloader child/parent
        if os.environ.get('WERKZEUG_RUN_MAIN') != 'true' and app.debug:
            return
        
        while True:
            time.sleep(interval_seconds)
            try:
                now = time.time()
                for filename in os.listdir(upload_folder):
                    file_path = os.path.join(upload_folder, filename)
                    if os.path.isfile(file_path):
                        file_modified = os.path.getmtime(file_path)
                        if now - file_modified > max_age_seconds:
                            os.remove(file_path)
            except Exception as e:
                print(f"Error in cleanup thread: {e}")
                
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()

start_cleanup_thread(app.config['UPLOAD_FOLDER'])

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_dir = os.path.dirname(os.path.abspath(__file__))
vgg_path = os.path.join(current_dir, 'vgg_normalised.pth')
decoder_path = os.path.join(current_dir, 'experiment', 'final_exp', 'decoder_final.pth')

encoder = VGGEncoder(vgg_path).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))

encoder.eval()
decoder.eval()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)



@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == 'POST':
        if form.validate_on_submit():
            # Resolve content image: uploaded file > preset > existing path
            if form.content.data and form.content.data.filename:
                if allowed_file(form.content.data.filename):
                    unique_prefix = uuid.uuid4().hex[:8]
                    content_filename = f"{unique_prefix}_{secure_filename(form.content.data.filename)}"
                    form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                    form.content_path.data = content_filename
            elif request.form.get('content_preset'):
                content_filename = secure_filename(request.form.get('content_preset'))
            else:
                content_filename = form.content_path.data

            # Resolve style image: uploaded file > preset > existing path
            if form.style.data and form.style.data.filename:
                if allowed_file(form.style.data.filename):
                    unique_prefix = uuid.uuid4().hex[:8]
                    style_filename = f"{unique_prefix}_{secure_filename(form.style.data.filename)}"
                    form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                    form.style_path.data = style_filename
            elif request.form.get('style_preset'):
                style_filename = secure_filename(request.form.get('style_preset'))
            else:
                style_filename = form.style_path.data

            if content_filename and style_filename:
                # Resolve paths: uploads folder for user files, content_data/style_data/examples for presets
                content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
                if not os.path.isfile(content_path):
                    for folder in ['content_data', 'examples']:
                        candidate = os.path.join(current_dir, folder, content_filename)
                        if os.path.isfile(candidate):
                            content_path = candidate
                            break

                style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
                if not os.path.isfile(style_path):
                    for folder in ['style_data', 'examples']:
                        candidate = os.path.join(current_dir, folder, style_filename)
                        if os.path.isfile(candidate):
                            style_path = candidate
                            break
                
                try:
                    if not os.path.isfile(content_path):
                        raise FileNotFoundError(f"Content image not found: {content_filename}")
                    if not os.path.isfile(style_path):
                        raise FileNotFoundError(f"Style image not found: {style_filename}")

                    # File size check (max 10MB)
                    if os.path.getsize(content_path) > 10 * 1024 * 1024 or os.path.getsize(style_path) > 10 * 1024 * 1024:
                        raise ValueError("Images must be smaller than 10MB.")

                    content_image = Image.open(content_path).convert('RGB')
                    style_image = Image.open(style_path).convert('RGB')

                    alpha = float(form.alpha.data)
                    stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

                    unique_prefix = uuid.uuid4().hex[:8]
                    result_filename = f"stylized_{unique_prefix}_{content_filename}"
                    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                    save_image(stylized_image, result_path)
                    
                    result_image = result_filename
                except Exception as e:
                    error = str(e)
                    # Clean up invalid/corrupted files immediately on failure
                    for p in [content_path, style_path]:
                        if os.path.exists(p) and app.config['UPLOAD_FOLDER'] in p:
                            try:
                                os.remove(p)
                            except:
                                pass
            else:
                if not content_filename:
                    error = 'Please upload content image'
                elif not style_filename:
                    error = 'Please upload style image'
        else:
            error = 'Form validation failed. Please check your inputs.'

    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
                           style_image=style_filename, error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


@app.route('/favicon.ico')
def favicon():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="12" fill="#17212f"/><path d="M18 42 32 14l14 28H18Z" fill="#b88715"/><circle cx="32" cy="42" r="9" fill="#126c6a"/></svg>"""
    response = app.response_class(svg, mimetype='image/svg+xml')
    response.headers['Cache-Control'] = 'public, max-age=86400'
    return response


@app.route('/api/transfer', methods=['POST'])
def api_transfer():
    content_file = request.files.get('content')
    style_file = request.files.get('style')
    
    content_preset = request.form.get('content_preset')
    style_preset = request.form.get('style_preset')
    
    alpha_val = request.form.get('alpha', 1.0)
    
    try:
        alpha = float(alpha_val)
        if not (0.0 <= alpha <= 1.0):
            return jsonify({"success": False, "error": "Alpha must be between 0.0 and 1.0."}), 400
    except ValueError:
        return jsonify({"success": False, "error": "Invalid alpha value."}), 400

    content_path = None
    style_path = None
    files_to_cleanup = []

    # 1. Resolve content image
    if content_file and content_file.filename != '':
        if not allowed_file(content_file.filename):
            return jsonify({"success": False, "error": "Unsupported content file format."}), 400
        content_filename = secure_filename(content_file.filename)
        unique_suffix = uuid.uuid4().hex[:8]
        content_filename = f"{unique_suffix}_{content_filename}"
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
        try:
            content_file.save(content_path)
            files_to_cleanup.append(content_path)
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to save content image: {e}"}), 500
    elif content_preset:
        preset_name = secure_filename(content_preset)
        search_paths = [
            os.path.join(current_dir, 'content_data', preset_name),
            os.path.join(current_dir, 'examples', preset_name)
        ]
        for path in search_paths:
            if os.path.isfile(path):
                content_path = path
                break
        if not content_path:
            return jsonify({"success": False, "error": f"Content preset '{preset_name}' not found."}), 400
    else:
        return jsonify({"success": False, "error": "Content image is required (upload a file or select a preset)."}), 400

    # 2. Resolve style image
    if style_file and style_file.filename != '':
        if not allowed_file(style_file.filename):
            for p in files_to_cleanup:
                if os.path.exists(p):
                    os.remove(p)
            return jsonify({"success": False, "error": "Unsupported style file format."}), 400
        style_filename = secure_filename(style_file.filename)
        unique_suffix = uuid.uuid4().hex[:8]
        style_filename = f"{unique_suffix}_{style_filename}"
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
        try:
            style_file.save(style_path)
            files_to_cleanup.append(style_path)
        except Exception as e:
            for p in files_to_cleanup:
                if os.path.exists(p):
                    os.remove(p)
            return jsonify({"success": False, "error": f"Failed to save style image: {e}"}), 500
    elif style_preset:
        preset_name = secure_filename(style_preset)
        search_paths = [
            os.path.join(current_dir, 'style_data', preset_name),
            os.path.join(current_dir, 'examples', preset_name)
        ]
        for path in search_paths:
            if os.path.isfile(path):
                style_path = path
                break
        if not style_path:
            for p in files_to_cleanup:
                if os.path.exists(p):
                    os.remove(p)
            return jsonify({"success": False, "error": f"Style preset '{preset_name}' not found."}), 400
    else:
        for p in files_to_cleanup:
            if os.path.exists(p):
                os.remove(p)
        return jsonify({"success": False, "error": "Style image is required (upload a file or select a preset)."}), 400

    # 3. Perform Style Transfer
    try:
        if os.path.getsize(content_path) > 10 * 1024 * 1024 or os.path.getsize(style_path) > 10 * 1024 * 1024:
            raise ValueError("Images must be smaller than 10MB.")

        content_image = Image.open(content_path).convert('RGB')
        style_image = Image.open(style_path).convert('RGB')

        stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

        base_name = os.path.basename(content_path)
        unique_suffix = uuid.uuid4().hex[:8]
        result_filename = f"stylized_{unique_suffix}_{base_name}"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        save_image(stylized_image, result_path)

        return jsonify({
            "success": True,
            "result_url": url_for('send_image', filename=result_filename)
        })
        
    except Exception as e:
        for p in files_to_cleanup:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True, threaded=True)





