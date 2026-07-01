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

from unittest.mock import MagicMock, patch
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

