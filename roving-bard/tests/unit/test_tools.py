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

from unittest.mock import MagicMock, patch, mock_open
from PIL import Image
from app import tools


def test_model_name_translation():
    """Verify call_gemini_vision translates deprecated model names to gemini-2.5-flash-lite."""
    img = Image.new("RGB", (100, 100))
    
    with patch("litellm.completion") as mock_completion:
        mock_choice = MagicMock()
        mock_choice.message.content = '{"location": "Tinnudir", "coordinates": "11.9S, 67.8W"}'
        mock_completion.return_value.choices = [mock_choice]
        
        # Call with legacy 1.5 flash name
        tools.call_gemini_vision(img, "gemini/gemini-1.5-flash")
        
        # Verify it translated and called litellm with 2.5 flash-lite
        _, called_kwargs = mock_completion.call_args
        assert called_kwargs["model"] == "gemini/gemini-2.5-flash-lite"


def test_json_boundary_extraction_robustness():
    """Verify call_gemini_vision can parse JSON even with leading/trailing markdown/extra characters."""
    img = Image.new("RGB", (100, 100))
    
    dirty_responses = [
        '```json\n{"location": "Tinnudir", "coordinates": "11.9S, 67.8W"}\n```',
        'Here is the JSON object:\n{"location": "Tinnudir", "coordinates": "11.9S, 67.8W"}\nHope this helps!',
        '{"location": "Tinnudir", "coordinates": "11.9S, 67.8W"} trailing details',
        '\n\n{"location": "Tinnudir", "coordinates": "11.9S, 67.8W"}\n\n'
    ]
    
    for resp in dirty_responses:
        with patch("litellm.completion") as mock_completion:
            mock_choice = MagicMock()
            mock_choice.message.content = resp
            mock_completion.return_value.choices = [mock_choice]
            
            loc, coords, ns, ew = tools.call_gemini_vision(img, "gemini/gemini-2.5-flash-lite")
            
            assert loc == "Tinnudir"
            assert coords == "11.9S, 67.8W"
            assert ns == -11.9
            assert ew == -67.8


def test_parse_text_same_line():
    """Verify parse_text extracts both coordinates and location even if on the same line."""
    from app.player import LocalOCRParser
    
    raw_texts = [
        "Tinnudir 11.9S, 67.8W",
        "11.9S, 67.8W Tinnudir",
        "Tinnudir\n11.9S, 67.8W",
        "11.9S, 67.8W\nTinnudir"
    ]
    
    for text in raw_texts:
        loc, coords, ns, ew = LocalOCRParser.parse_text(text)
        assert loc == "Tinnudir"
        assert coords == "11.9S, 67.8W"
        assert ns == -11.9
        assert ew == -67.8

    # VLM specific circle border noise cases
    vlm_noise_texts = [
        "xtrinudir 11.9S, 67.8W",
        "xtnudir 11.9S, 67.8W",
        "xtractable Tinnudir 11.9S, 67.8W"
    ]
    for text in vlm_noise_texts:
        loc, coords, ns, ew = LocalOCRParser.parse_text(text)
        assert loc == "Tinnudir"
        assert coords == "11.9S, 67.8W"


def test_parse_text_rich():
    """Verify parse_text_rich returns correct parsed and raw unfuzzy values."""
    from app.player import LocalOCRParser
    
    # Check standard VLM response with noise
    rich = LocalOCRParser.parse_text_rich("xtrinudir 11.9S, 67.8W")
    assert rich["parsed_location"] == "Tinnudir"
    assert rich["raw_location"] == "xtrinudir"
    assert rich["parsed_coordinates"] == "11.9S, 67.8W"
    assert rich["raw_coordinates"] == "11.9S, 67.8W"
    
    # Check Moondream warmup failure token
    rich2 = LocalOCRParser.parse_text_rich("xtrp")
    assert rich2["raw_location"] == "xtrp"
    assert rich2["parsed_location"] in ("xtrp", "p", "None") # cleaned is "xtrp" -> "p", or "None" if filtered
    assert rich2["parsed_coordinates"] == "None"
    assert rich2["raw_coordinates"] == "None"


def test_simulation_mode_minimap_bounds_detection():
    """Verify that in simulation mode, force_manual_bounds is ignored, auto-detection runs,

    and it falls back to manual bounds appropriately based on config.yaml.
    """
    from app.fast_api_app import (
        start_async_minimap_detection,
        is_simulation_mode,
        has_minimap_bounds_in_yaml,
    )
    from app import tools

    class SyncThread:
        def __init__(self, target, daemon=True):
            self.target = target
        def start(self):
            self.target()

    # Create a dummy image
    img = Image.new("RGB", (100, 100))

    # Scenario 1: Verify is_simulation_mode matches correctly
    with patch("os.path.exists", return_value=True), \
         patch("os.listdir", return_value=["test_screenshot1.png", "not_test.png", "test_screenshot2.jpg"]):
        assert is_simulation_mode() is True

    with patch("os.path.exists", return_value=True), \
         patch("os.listdir", return_value=["not_test.png"]):
        assert is_simulation_mode() is False

    # Scenario 2: Verify has_minimap_bounds_in_yaml checks the raw YAML file correctly
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data="minimap_bounds:\n  x: 0.1\n")):
        assert has_minimap_bounds_in_yaml() is True

    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data="force_manual_bounds: false\n")):
        assert has_minimap_bounds_in_yaml() is False

    # Scenario 3: Verify start_async_minimap_detection in simulation mode
    with patch("threading.Thread", SyncThread), \
         patch("app.fast_api_app.is_simulation_mode", return_value=True), \
         patch("app.tools.check_screen_and_update_music") as mock_check_screen:
        
        # Test Case A: Detection succeeds in Simulation Mode -> uses detected bounds
        with patch("app.tools.load_config", return_value={"force_manual_bounds": True}), \
             patch("app.fast_api_app.has_minimap_bounds_in_yaml", return_value=True), \
             patch.object(tools.grabber, "detect_minimap", return_value=({"x": 0.5, "y": 0.5, "width": 0.2, "height": 0.2}, True)):
            
            start_async_minimap_detection(img)
            
            # Since is_sim=True, force_manual_bounds should be ignored (auto-detection runs)
            assert tools.minimap_detected is True
            assert tools.grabber.bounds == {"x": 0.5, "y": 0.5, "width": 0.2, "height": 0.2}

        # Test Case B: Detection fails, minimap_bounds exists in yaml -> falls back to config's minimap_bounds
        with patch("app.tools.load_config", return_value={"force_manual_bounds": True, "minimap_bounds": {"x": 0.7, "y": 0.7, "width": 0.1, "height": 0.1}}), \
             patch("app.fast_api_app.has_minimap_bounds_in_yaml", return_value=True), \
             patch.object(tools.grabber, "detect_minimap", return_value=(None, False)):
            
            start_async_minimap_detection(img)
            
            assert tools.minimap_detected is False
            assert tools.grabber.bounds == {"x": 0.7, "y": 0.7, "width": 0.1, "height": 0.1}

        # Test Case C: Detection fails, minimap_bounds does NOT exist in yaml -> falls back to default bounds
        with patch("app.tools.load_config", return_value={"force_manual_bounds": True}), \
             patch("app.fast_api_app.has_minimap_bounds_in_yaml", return_value=False), \
             patch.object(tools.grabber, "detect_minimap", return_value=(None, False)):
            
            start_async_minimap_detection(img)
            
            assert tools.minimap_detected is False
            assert tools.grabber.bounds == {"x": 0.8, "y": 0.05, "width": 0.15, "height": 0.15}


