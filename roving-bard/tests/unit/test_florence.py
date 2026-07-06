# Copyright 2026 Google LLC
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

import io
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from app.fast_api_app import (
    load_florence_model,
    run_florence_ocr,
    api_ocr_try_vlm,
    VlmTryRequest,
    vlm_download_states,
)


@pytest.fixture(autouse=True)
def reset_florence_globals():
    """Resets the global florence_model and florence_processor before each test."""
    from app import fast_api_app
    fast_api_app.florence_model = None
    fast_api_app.florence_processor = None
    yield
    fast_api_app.florence_model = None
    fast_api_app.florence_processor = None


def test_load_florence_model_cpu():
    """Verify load_florence_model initializes Florence-2 on CPU when CUDA/MPS are unavailable."""
    mock_model = MagicMock()
    mock_processor = MagicMock()

    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_from_pretrained, \
         patch("transformers.AutoProcessor.from_pretrained", return_value=mock_processor) as mock_proc_from_pretrained, \
         patch("app.fast_api_app.check_memory_safety") as mock_mem_safety:

        load_florence_model()

        # Check model loaded on CPU
        mock_from_pretrained.assert_called_once()
        mock_proc_from_pretrained.assert_called_once()
        mock_model.to.assert_called_with("cpu")


def test_load_florence_model_cuda():
    """Verify load_florence_model initializes Florence-2 on GPU when CUDA is available."""
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_model.to.return_value = mock_model

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_capability", return_value=(6, 1)), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_from_pretrained, \
         patch("transformers.AutoProcessor.from_pretrained", return_value=mock_processor) as mock_proc_from_pretrained, \
         patch("app.fast_api_app.check_memory_safety") as mock_mem_safety:

        load_florence_model()

        # Check model loaded on CUDA
        mock_from_pretrained.assert_called_once()
        mock_model.to.assert_called_with("cuda")


def test_run_florence_ocr():
    """Verify run_florence_ocr preprocesses image, feeds it to model, and decodes response."""
    mock_model = MagicMock()
    mock_processor = MagicMock()
    
    mock_model.device = MagicMock()
    mock_model.device.type = "cpu"
    mock_model.dtype = "float32"
    mock_model.to.return_value = mock_model
    mock_model.float.return_value = mock_model
    
    # Mock generation output IDs
    mock_generated_ids = [1, 2, 3]
    mock_model.generate.return_value = mock_generated_ids
    
    # Mock processor encoding and decoding
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_inputs.items.return_value = [("input_ids", MagicMock()), ("pixel_values", MagicMock())]
    mock_processor.return_value = mock_inputs
    mock_processor.batch_decode.return_value = ["Tinnudir [11.9S, 67.8W]"]
    mock_processor.post_process_generation.return_value = {"<OCR>": "Tinnudir [11.9S, 67.8W]"}

    img = Image.new("RGB", (100, 50), color="white")

    with patch("torch.cuda.is_available", return_value=False), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
         patch("transformers.AutoProcessor.from_pretrained", return_value=mock_processor), \
         patch("app.fast_api_app.check_memory_safety"):

        # Run OCR
        res_text = run_florence_ocr(img)

        assert res_text == "Tinnudir [11.9S, 67.8W]"
        mock_model.generate.assert_called_once()
        mock_processor.batch_decode.assert_called_with(mock_generated_ids, skip_special_tokens=True)


def test_api_ocr_try_vlm_florence():
    """Verify api_ocr_try_vlm successfully calls Florence-2 pipeline and parses response."""
    from app import tools
    original_raw = tools.latest_location_raw_bytes
    original_screenshot = tools.latest_screenshot_bytes
    original_detecting = tools.minimap_detecting
    
    # Create small valid mock PNG image
    img = Image.new("RGB", (100, 50), color="white")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    dummy_png_bytes = buffered.getvalue()
    
    tools.latest_location_raw_bytes = dummy_png_bytes
    tools.latest_screenshot_bytes = dummy_png_bytes
    tools.minimap_detecting = False
    
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_model.device = MagicMock()
    mock_model.device.type = "cpu"
    mock_model.dtype = "float32"
    mock_model.generate.return_value = [1, 2, 3]
    mock_model.to.return_value = mock_model
    mock_model.float.return_value = mock_model
    
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_inputs.items.return_value = [("input_ids", MagicMock()), ("pixel_values", MagicMock())]
    mock_processor.return_value = mock_inputs
    mock_processor.batch_decode.return_value = ["Tinnudir 11.9S, 67.8W"]
    mock_processor.post_process_generation.return_value = {"<OCR>": "Tinnudir 11.9S, 67.8W"}

    try:
        vlm_download_states["florence-2"]["ready"] = True

        with patch("torch.cuda.is_available", return_value=False), \
             patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
             patch("transformers.AutoProcessor.from_pretrained", return_value=mock_processor), \
             patch("app.fast_api_app.check_memory_safety"):

            req = VlmTryRequest(model="florence-2")
            res = api_ocr_try_vlm(req)

            assert res["status"] == "success"
            assert res["model"] == "Florence-2 (Large)"
            assert res["parsed_location"] == "Tinnudir"
            assert res["parsed_coordinates"] == "11.9S, 67.8W"
    finally:
        tools.latest_location_raw_bytes = original_raw
        tools.latest_screenshot_bytes = original_screenshot
        tools.minimap_detecting = original_detecting
