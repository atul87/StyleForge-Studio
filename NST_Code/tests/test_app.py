import os
import sys
import unittest
import io
import json
import shutil
from unittest.mock import patch

# Ensure the parent directory (NST_Code) is in the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app import app

class TestApp(unittest.TestCase):
    def setUp(self):
        # Configure app for testing
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['UPLOAD_FOLDER'] = os.path.join(parent_dir, 'static', 'test_uploads')
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        self.client = app.test_client()

    def tearDown(self):
        # Clean up test uploads directory
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            shutil.rmtree(app.config['UPLOAD_FOLDER'])

    def test_index_get(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        # Check some expected elements in templates/index.html
        self.assertIn('StyleForge', html)
        self.assertIn('Content Image', html)
        self.assertIn('Style Image', html)

    def test_favicon(self):
        response = self.client.get('/favicon.ico')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'image/svg+xml')

    def test_api_transfer_presets_success(self):
        # Test transfer using valid presets (brad_pitt.jpg and sketch.png)
        # These presets exist in content_data and style_data respectively.
        response = self.client.post('/api/transfer', data={
            'content_preset': 'brad_pitt.jpg',
            'style_preset': 'sketch.png',
            'alpha': '0.8'
        })
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertTrue(data['success'])
        self.assertIn('result_url', data)
        self.assertTrue(data['result_url'].startswith('/uploads/stylized_'))

        # Verify the created result file exists
        result_filename = os.path.basename(data['result_url'])
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        self.assertTrue(os.path.isfile(result_path))

    def test_api_transfer_file_upload_success(self):
        # Create dummy image data
        content_img = io.BytesIO()
        from PIL import Image
        Image.new('RGB', (100, 100), color='white').save(content_img, 'JPEG')
        content_img.seek(0)

        style_img = io.BytesIO()
        Image.new('RGB', (100, 100), color='blue').save(style_img, 'PNG')
        style_img.seek(0)

        response = self.client.post('/api/transfer', data={
            'content': (content_img, 'uploaded_content.jpg'),
            'style': (style_img, 'uploaded_style.png'),
            'alpha': '0.5'
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertTrue(data['success'])
        self.assertIn('result_url', data)

    def test_api_transfer_invalid_alpha(self):
        # Alpha > 1.0 should return 400
        response1 = self.client.post('/api/transfer', data={
            'content_preset': 'brad_pitt.jpg',
            'style_preset': 'sketch.png',
            'alpha': '1.5'
        })
        self.assertEqual(response1.status_code, 400)
        data1 = json.loads(response1.data.decode('utf-8'))
        self.assertFalse(data1['success'])
        self.assertIn('Alpha must be between 0.0 and 1.0', data1['error'])

        # Alpha non-float should return 400
        response2 = self.client.post('/api/transfer', data={
            'content_preset': 'brad_pitt.jpg',
            'style_preset': 'sketch.png',
            'alpha': 'invalid_string'
        })
        self.assertEqual(response2.status_code, 400)
        data2 = json.loads(response2.data.decode('utf-8'))
        self.assertFalse(data2['success'])
        self.assertIn('Invalid alpha value', data2['error'])

    def test_api_transfer_unsupported_file_format(self):
        content_txt = io.BytesIO(b"not an image file")
        response = self.client.post('/api/transfer', data={
            'content': (content_txt, 'dummy.txt'),
            'style_preset': 'sketch.png'
        }, content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data.decode('utf-8'))
        self.assertFalse(data['success'])
        self.assertIn('Unsupported content file format', data['error'])

    def test_api_transfer_missing_parameters(self):
        # Missing content should fail
        response = self.client.post('/api/transfer', data={
            'style_preset': 'sketch.png'
        })
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data.decode('utf-8'))
        self.assertFalse(data['success'])
        self.assertIn('Content image is required', data['error'])

    @patch('os.path.getsize')
    def test_api_transfer_file_too_large(self, mock_getsize):
        # Force getsize to return more than 10MB
        mock_getsize.return_value = 11 * 1024 * 1024  
        
        response = self.client.post('/api/transfer', data={
            'content_preset': 'brad_pitt.jpg',
            'style_preset': 'sketch.png'
        })
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data.decode('utf-8'))
        self.assertFalse(data['success'])
        self.assertIn('Images must be smaller than 10MB', data['error'])

    def test_ui_transfer_post_success(self):
        response = self.client.post('/', data={
            'content_preset': 'brad_pitt.jpg',
            'style_preset': 'sketch.png',
            'alpha': 1.0,
            'submit': True
        })
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        # Check if the result image is rendered in the HTML response
        self.assertIn('stylized_', html)

if __name__ == '__main__':
    unittest.main()
