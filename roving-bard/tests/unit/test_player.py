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

from app.player import LocalOCRParser, TrackMapper


def test_coordinate_parsing() -> None:
    # Test North/West coordinate string parsing
    loc, coords, ns, ew = LocalOCRParser.parse_text("Town\n19.3N, 70.9W")
    assert loc == "Town"
    assert coords == "19.3N, 70.9W"
    assert ns == 19.3
    assert ew == -70.9

    # Test South/East coordinate string parsing
    loc, coords, ns, ew = LocalOCRParser.parse_text("Forest\n14.9S, 103.1E")
    assert loc == "Forest"
    assert coords == "14.9S, 103.1E"
    assert ns == -14.9
    assert ew == 103.1

    # Test parser handling noise/symbols
    loc, coords, ns, ew = LocalOCRParser.parse_text(
        "!!! [Area] Boss Arena !!!\n42.0N - 12.5E"
    )
    assert loc == "Area Boss Arena"
    assert coords == "42.0N, 12.5E"
    assert ns == 42.0
    assert ew == 12.5
    # Reset state to default for testing
    LocalOCRParser._last_lat_dir = "S"
    LocalOCRParser._last_lon_dir = "W"
    # Test partial coordinate parsing (missing latitude direction letter, should fallback to cached/default 'S')
    loc, coords, ns, ew = LocalOCRParser.parse_text("Tinnudir\n11.9, 67.8W")
    assert loc == "Tinnudir"
    assert coords == "11.9S, 67.8W"
    assert ns == -11.9
    assert ew == -67.8

    # Test partial coordinate parsing (missing longitude direction letter, should fallback to cached/default 'W')
    loc, coords, ns, ew = LocalOCRParser.parse_text("Tinnudir\n11.9S, 67.8")
    assert loc == "Tinnudir"
    assert coords == "11.9S, 67.8W"
    assert ns == -11.9
    assert ew == -67.8
    # Test prose cleaning extraction and fuzzy matching to Tinnudir
    loc, coords, ns, ew = LocalOCRParser.parse_text(
        "The image shows a wooden door. In the center, there are two lines of text that read Tinnuur 11.9S, 67.8W"
    )
    assert loc == "Tinnudir"
    assert coords == "11.9S, 67.8W"

    # Test separated coordinates (Run 14 case: '11.9S', and '67.8W')
    loc, coords, ns, ew = LocalOCRParser.parse_text(
        "In the top left corner of the image, there are some text elements, including 'Tinnudir', '11.9S', and '67.8W'."
    )
    assert loc == "Tinnudir"
    assert coords == "11.9S, 67.8W"
    assert ns == -11.9
    assert ew == -67.8


def test_track_mapping() -> None:
    mappings = [
        {"location_name": "Town", "track_file": "town.wav"},
        {"location_name": "Forest", "track_file": "forest.wav"},
        {
            "ns_min": 10.0,
            "ns_max": 20.0,
            "ew_min": -80.0,
            "ew_max": -60.0,
            "track_file": "cave.wav",
        },
    ]
    mapper = TrackMapper(mappings)

    # Test location name match
    assert (
        mapper.get_track_for_state("Welcome to Town Center", None, None) == "town.wav"
    )
    assert mapper.get_track_for_state("Deep Forest Region", None, None) == "forest.wav"

    # Test coordinate range match
    assert mapper.get_track_for_state(None, 15.0, -70.0) == "cave.wav"  # Inside bounds
    assert mapper.get_track_for_state(None, 25.0, -70.0) is None  # Outside NS bounds
    assert mapper.get_track_for_state(None, 15.0, -90.0) is None  # Outside EW bounds

    # Location name takes priority
    assert mapper.get_track_for_state("Town Center", 15.0, -70.0) == "town.wav"
