# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import time
import io
from unittest.mock import patch, MagicMock
from PIL import Image
import pytest

from app.fast_api_app import (
    api_ocr_try_vlm,
    api_gc,
    VlmTryRequest,
    GcRequest,
    vlm_inference_cancel_event,
    vlm_download_states,
)


def test_vlm_cancel_via_gc() -> None:
    """Test that invoking the GC endpoint cancels a running model inference in try_vlm."""
    enter_post_event = threading.Event()
    finish_post_event = threading.Event()
    response_mock = MagicMock()
    
    # Mocking iter_lines to simulate a streaming/blocking request that checks cancel event
    def mock_iter_lines():
        # Set event indicating we have entered the requests call
        enter_post_event.set()
        # Wait up to 5 seconds for GC to set the cancel event
        for _ in range(50):
            if vlm_inference_cancel_event.is_set():
                break
            time.sleep(0.1)
        finish_post_event.set()
        yield b'{"response": "dummy", "done": true}'
        
    response_mock.status_code = 200
    response_mock.iter_lines = mock_iter_lines
    
    get_mock = MagicMock()
    get_mock.status_code = 200
    get_mock.json.return_value = {"models": []}
    
    # Mock tools.latest_location_raw_bytes and tools.latest_screenshot_bytes to not trigger screen scan
    from app import tools
    original_raw = tools.latest_location_raw_bytes
    original_screenshot = tools.latest_screenshot_bytes
    
    # Create a small valid mock PNG image
    img = Image.new("RGB", (100, 50), color="white")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    dummy_png_bytes = buffered.getvalue()
    
    tools.latest_location_raw_bytes = dummy_png_bytes
    tools.latest_screenshot_bytes = dummy_png_bytes
    
    try_vlm_result = {}
    
    def run_try_vlm():
        try:
            # We request a local model like moondream
            # We mock ready state of moondream so it doesn't fail on ready check
            vlm_download_states["moondream"]["ready"] = True
            
            req = VlmTryRequest(model="moondream")
            res = api_ocr_try_vlm(req)
            try_vlm_result["res"] = res
        except Exception as e:
            try_vlm_result["error"] = str(e)
            
    with patch("requests.get", return_value=get_mock) as mock_get, \
         patch("requests.post", return_value=response_mock) as mock_post, \
         patch("app.fast_api_app.check_memory_safety") as mock_memory_safety:
        t = threading.Thread(target=run_try_vlm)
        t.start()
        
        # Wait for the thread to enter requests.post
        assert enter_post_event.wait(timeout=5)
        
        # Now trigger the GC call
        gc_req = GcRequest(model="moondream")
        gc_res = api_gc(gc_req)
        assert gc_res["status"] == "success"
        
        # Wait for the try_vlm thread to finish
        t.join(timeout=5)
        
        # Restore tools attributes
        tools.latest_location_raw_bytes = original_raw
        tools.latest_screenshot_bytes = original_screenshot
        
        # Check that try_vlm returned a cancelled/error response
        assert "res" in try_vlm_result
        assert try_vlm_result["res"]["status"] == "error"
        assert "cancelled" in try_vlm_result["res"]["message"].lower()


def test_vlm_image_size_guard() -> None:
    """Ensure that too large location crops are rejected with a helpful error message."""
    from app import tools
    original_raw = tools.latest_location_raw_bytes
    original_screenshot = tools.latest_screenshot_bytes
    
    # Create an image that is too large (e.g. 1200x600)
    img = Image.new("RGB", (1200, 600), color="white")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    large_png_bytes = buffered.getvalue()
    
    tools.latest_location_raw_bytes = large_png_bytes
    tools.latest_screenshot_bytes = large_png_bytes
    
    try:
        req = VlmTryRequest(model="moondream")
        res = api_ocr_try_vlm(req)
        assert res["status"] == "error"
        assert "too large" in res["message"].lower()
    finally:
        tools.latest_location_raw_bytes = original_raw
        tools.latest_screenshot_bytes = original_screenshot


def test_vlm_memory_safety_guard() -> None:
    """Ensure that low memory triggers an error response to protect the host."""
    from app import tools
    original_raw = tools.latest_location_raw_bytes
    original_screenshot = tools.latest_screenshot_bytes
    
    # Create a small valid image
    img = Image.new("RGB", (100, 50), color="white")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    dummy_png_bytes = buffered.getvalue()
    
    tools.latest_location_raw_bytes = dummy_png_bytes
    tools.latest_screenshot_bytes = dummy_png_bytes
    
    # Mock system memory to be very low (e.g. 100 MB available)
    with patch("app.fast_api_app.get_available_system_ram_bytes", return_value=100*1024*1024):
        try:
            req = VlmTryRequest(model="moondream")
            res = api_ocr_try_vlm(req)
            assert res["status"] == "error"
            assert "insufficient system ram" in res["message"].lower()
        finally:
            tools.latest_location_raw_bytes = original_raw
            tools.latest_screenshot_bytes = original_screenshot
