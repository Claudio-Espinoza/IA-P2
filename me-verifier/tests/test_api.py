"""
Test suite for Flask API
"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.app import app


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_healthz(client):
    """Test health check endpoint"""
    response = client.get('/healthz')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert data['status'] == 'healthy'


def test_verify_no_image(client):
    """Test verify endpoint without image"""
    response = client.post('/verify')
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data


def test_verify_with_image(client):
    """Test verify endpoint with image"""
    # Create a dummy image file
    from io import BytesIO
    from PIL import Image
    
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    response = client.post('/verify',
                          data={'image': (img_bytes, 'test.jpg')},
                          content_type='multipart/form-data')
    
    assert response.status_code in [200, 503]  # 503 if model not loaded
    data = response.get_json()
    
    if response.status_code == 200:
        assert 'is_me' in data or 'message' in data
