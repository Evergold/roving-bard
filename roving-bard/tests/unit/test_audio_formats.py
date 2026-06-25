import os
from fastapi.testclient import TestClient

from app.fast_api_app import app
from app import tools

client = TestClient(app)

def get_headers():
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("AGENT_API_KEY")
        or "test-mock-key"
    )
    # Ensure the key is allowed by tools.config
    tools.config["api_key"] = api_key
    return {"X-API-Key": api_key}


def test_supported_audio_formats(tmp_path):
    """Test that all supported audio formats are accepted."""
    
    # Save original playlist dir and temporarily override to avoid cluttering workspace
    original_playlist_dir = tools.player.playlist_dir
    tools.player.playlist_dir = str(tmp_path)
    
    supported_extensions = [".wav", ".mp3", ".ogg", ".flac", ".abc", ".mp4"]
    
    try:
        for ext in supported_extensions:
            filename = f"test_audio{ext}"
            file_payload = b"dummy content"
            
            headers = {"X-API-Key": get_headers()["X-API-Key"]}
            
            response = client.post(
                "/api/upload-audio",
                headers=headers,
                files={"file": (filename, file_payload, "audio/mpeg")}
            )
            
            assert response.status_code == 200, f"Expected 200 for {ext}, got {response.status_code}. Response: {response.text}"
            assert response.json()["status"] == "success"
            
    finally:
        tools.player.playlist_dir = original_playlist_dir


def test_unsupported_audio_format(tmp_path):
    """Test that unsupported formats are rejected."""
    
    original_playlist_dir = tools.player.playlist_dir
    tools.player.playlist_dir = str(tmp_path)
    
    try:
        filename = "test_audio.txt"
        file_payload = b"dummy content"
        
        headers = {"X-API-Key": get_headers()["X-API-Key"]}
        
        response = client.post(
            "/api/upload-audio",
            headers=headers,
            files={"file": (filename, file_payload, "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Unsupported audio file format" in response.json()["detail"]
            
    finally:
        tools.player.playlist_dir = original_playlist_dir
