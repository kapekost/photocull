import json
from datetime import datetime
from io import StringIO

from rich.console import Console

from photocull.report import render_table, summary_to_json_dict, write_json
from photocull.summary import build_summary

from .fixtures import make_photo_record


def _sample_summary():
    records = [
        make_photo_record(date=datetime(2025, 6, 1), filesize=100, is_screenshot=True),
        make_photo_record(
            date=datetime(2025, 6, 1), filesize=200, is_video=True, duration_seconds=3
        ),
        make_photo_record(date=datetime(2024, 1, 1), filesize=50, is_favorite=True),
    ]
    return build_summary(records, library_path="/fake/lib.photoslibrary")


def test_summary_to_json_dict_is_plain_json_serializable():
    summary = _sample_summary()
    data = summary_to_json_dict(summary)
    # must not raise -- proves every value is a plain JSON type
    encoded = json.dumps(data)
    decoded = json.loads(encoded)
    assert decoded["total_items"] == 3
    assert decoded["total_photos"] == 2
    assert decoded["total_videos"] == 1
    assert decoded["by_year"]["2024"] == {"count": 1, "size_bytes": 50}
    assert decoded["by_type"]["screenshot"] == {"count": 1, "size_bytes": 100}
    assert decoded["videos_by_duration"]["<5s"] == {"count": 1, "size_bytes": 200}
    assert decoded["favorites_count"] == 1
    assert decoded["library_path"] == "/fake/lib.photoslibrary"
    assert isinstance(decoded["generated_at"], str)


def test_write_json_writes_the_same_shape_to_disk(tmp_path):
    summary = _sample_summary()
    out_path = tmp_path / "audit.json"
    write_json(summary, out_path)
    with open(out_path) as f:
        data = json.load(f)
    assert data["total_items"] == 3


def test_render_table_prints_key_figures():
    summary = _sample_summary()
    buffer = StringIO()
    console = Console(file=buffer, width=120)
    render_table(summary, console=console)
    output = buffer.getvalue()
    assert "3" in output  # total items
    assert "screenshot" in output.lower()
    assert "2024" in output
    assert "2025" in output
