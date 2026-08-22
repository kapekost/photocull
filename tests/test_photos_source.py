from datetime import datetime
from types import SimpleNamespace

from photocull.photos_source import iter_photo_records, keeper_uuids, record_from_photoinfo


def make_fake_photoinfo(**overrides):
    """A duck-typed stand-in for osxphotos.PhotoInfo -- same attribute surface, no
    osxphotos/Photos-library dependency. Lets us test the adapter's field mapping
    without needing Full Disk Access or a real library."""
    defaults = dict(
        uuid="fake-uuid-1",
        date=datetime(2025, 6, 1, 10, 0, 0),
        ismovie=False,
        original_filesize=12345,
        exif_info=None,
        screenshot=False,
        screen_recording=False,
        selfie=False,
        burst=False,
        live_photo=False,
        slow_mo=False,
        time_lapse=False,
        panorama=False,
        israw=False,
        favorite=False,
        hidden=False,
        shared=False,
        albums=[],
        burst_photos=[],
        intrash=False,
        latitude=None,
        longitude=None,
        date_modified=None,
        width=4032,
        height=3024,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_record_from_photoinfo_maps_every_field():
    p = make_fake_photoinfo(
        uuid="abc-123",
        date=datetime(2025, 6, 1),
        ismovie=False,
        original_filesize=999,
        screenshot=True,
        selfie=True,
        israw=True,
        favorite=True,
        hidden=True,
        albums=["Trip 2025", "Family"],
    )
    r = record_from_photoinfo(p)
    assert r.uuid == "abc-123"
    assert r.date == datetime(2025, 6, 1)
    assert r.is_video is False
    assert r.filesize == 999
    assert r.is_screenshot is True
    assert r.is_selfie is True
    assert r.is_raw is True
    assert r.is_favorite is True
    assert r.is_hidden is True
    assert r.album_count == 2


def test_record_from_photoinfo_duration_is_none_for_photos_even_with_exif_info():
    p = make_fake_photoinfo(ismovie=False, exif_info=SimpleNamespace(duration=5.0))
    r = record_from_photoinfo(p)
    assert r.duration_seconds is None


def test_record_from_photoinfo_duration_comes_from_exif_info_for_videos():
    p = make_fake_photoinfo(ismovie=True, exif_info=SimpleNamespace(duration=42.5))
    r = record_from_photoinfo(p)
    assert r.is_video is True
    assert r.duration_seconds == 42.5


def test_record_from_photoinfo_duration_is_none_when_exif_info_missing():
    # Photos <5 libraries: exif_info can be None even for a video
    p = make_fake_photoinfo(ismovie=True, exif_info=None)
    r = record_from_photoinfo(p)
    assert r.duration_seconds is None


def test_iter_photo_records_maps_every_item_from_the_db():
    fake_db = SimpleNamespace(
        photos=lambda: [
            make_fake_photoinfo(uuid="a"),
            make_fake_photoinfo(uuid="b"),
        ]
    )
    records = list(iter_photo_records(fake_db))
    assert [r.uuid for r in records] == ["a", "b"]


def test_iter_photo_records_includes_non_selected_burst_siblings():
    """PhotosDB.photos() returns only a burst group's key image and silently drops
    the non-selected siblings. Those siblings are exactly the near-duplicates this
    app exists to cull, so the adapter must expand them back in -- verified against
    the real library: 9 key images hid 72 siblings (14156 -> 14228 items)."""
    key = make_fake_photoinfo(
        uuid="key",
        burst=True,
        burst_photos=[
            make_fake_photoinfo(uuid="sib-1", burst=True),
            make_fake_photoinfo(uuid="sib-2", burst=True),
        ],
    )
    fake_db = SimpleNamespace(photos=lambda: [key])
    records = list(iter_photo_records(fake_db))
    assert [r.uuid for r in records] == ["key", "sib-1", "sib-2"]
    assert all(r.is_burst for r in records)


def test_iter_photo_records_does_not_double_count_a_sibling_photos_already_returned():
    shared = make_fake_photoinfo(uuid="sib", burst=True)
    key = make_fake_photoinfo(uuid="key", burst=True, burst_photos=[shared])
    fake_db = SimpleNamespace(photos=lambda: [key, shared])
    records = list(iter_photo_records(fake_db))
    assert [r.uuid for r in records] == ["key", "sib"]


def test_iter_photo_records_skips_trashed_burst_siblings():
    key = make_fake_photoinfo(
        uuid="key",
        burst=True,
        burst_photos=[make_fake_photoinfo(uuid="deleted-sib", burst=True, intrash=True)],
    )
    fake_db = SimpleNamespace(photos=lambda: [key])
    records = list(iter_photo_records(fake_db))
    assert [r.uuid for r in records] == ["key"]


def test_iter_photo_records_ignores_burst_photos_on_non_burst_items():
    p = make_fake_photoinfo(
        uuid="solo", burst=False, burst_photos=[make_fake_photoinfo(uuid="stray")]
    )
    fake_db = SimpleNamespace(photos=lambda: [p])
    records = list(iter_photo_records(fake_db))
    assert [r.uuid for r in records] == ["solo"]


def test_record_from_photoinfo_maps_phase1_fields():
    p = make_fake_photoinfo(
        latitude=37.9838,
        longitude=23.7275,
        date_modified=datetime(2025, 7, 4, 9, 30),
        width=4032,
        height=3024,
        exif_info=SimpleNamespace(duration=None, camera_model="iPhone 15 Pro"),
    )
    r = record_from_photoinfo(p)
    assert (r.latitude, r.longitude) == (37.9838, 23.7275)
    assert r.mod_date == datetime(2025, 7, 4, 9, 30)
    assert (r.width, r.height) == (4032, 3024)
    assert r.device == "iPhone 15 Pro"


def test_record_from_photoinfo_device_is_none_without_exif_info():
    r = record_from_photoinfo(make_fake_photoinfo(exif_info=None))
    assert r.device is None


def test_burst_group_members_all_share_one_synthetic_burst_key():
    """osxphotos has no usable public burst-group id: `PhotoInfo.burst_key` is a
    bool ('is this the key image') and `burst_albums` is empty on this library.
    Since iter_photo_records already expands the group, it stamps the key photo's
    uuid onto every member itself -- verified against Photos' own private
    burstUUID grouping. See DECISIONS.md burst-group-key-is-synthetic."""
    key = make_fake_photoinfo(
        uuid="key",
        burst=True,
        burst_photos=[
            make_fake_photoinfo(uuid="sib-1", burst=True),
            make_fake_photoinfo(uuid="sib-2", burst=True),
        ],
    )
    fake_db = SimpleNamespace(photos=lambda: [key])
    records = list(iter_photo_records(fake_db))
    assert [r.burst_key for r in records] == ["key", "key", "key"]


def test_separate_burst_groups_get_distinct_keys():
    g1 = make_fake_photoinfo(
        uuid="k1", burst=True, burst_photos=[make_fake_photoinfo(uuid="s1", burst=True)]
    )
    g2 = make_fake_photoinfo(
        uuid="k2", burst=True, burst_photos=[make_fake_photoinfo(uuid="s2", burst=True)]
    )
    fake_db = SimpleNamespace(photos=lambda: [g1, g2])
    by_uuid = {r.uuid: r.burst_key for r in iter_photo_records(fake_db)}
    assert by_uuid == {"k1": "k1", "s1": "k1", "k2": "k2", "s2": "k2"}


def test_non_burst_photos_have_no_burst_key():
    fake_db = SimpleNamespace(photos=lambda: [make_fake_photoinfo(uuid="solo")])
    assert [r.burst_key for r in iter_photo_records(fake_db)] == [None]


# --- derivative stamping (Task 8a) ---------------------------------------------


def test_derivative_path_is_not_resolved_unless_asked():
    """The audit path stays pure metadata. Resolving derivatives costs ~1.7 ms of
    ImageIO header reads per photo (~24 s across the real library) and Phase 0 has no
    use for them, so it must not happen by default."""
    p = make_fake_photoinfo(path_derivatives=["/thumbs/small.jpeg"])
    assert record_from_photoinfo(p).derivative_path is None


def test_stamps_the_chosen_derivative_when_asked():
    p = make_fake_photoinfo(path_derivatives=["/thumbs/small.jpeg"])
    r = record_from_photoinfo(p, with_derivative=True)
    assert r.derivative_path == "/thumbs/small.jpeg"


def test_a_photo_with_no_local_derivative_stamps_none():
    """Degrade to None; never escalate to the original (CLAUDE.md hard rule #2)."""
    p = make_fake_photoinfo(path_derivatives=[])
    assert record_from_photoinfo(p, with_derivative=True).derivative_path is None


def test_burst_siblings_get_their_derivative_stamped_too():
    """The load-bearing one. `PhotosDB.photos()` does not return burst siblings -- 72
    of them on the real library -- so this iteration is the only place their PhotoInfo
    is ever in hand. Resolving derivatives afterwards from a uuid lookup would leave
    exactly the near-duplicates this app exists to collapse with no path to pixels,
    and would do it silently. See DECISIONS.md `burst-siblings-expanded-in-scan`."""
    sibling = make_fake_photoinfo(
        uuid="sib-1", burst=True, path_derivatives=["/thumbs/sib.jpeg"]
    )
    key = make_fake_photoinfo(
        uuid="key-1",
        burst=True,
        burst_photos=[sibling],
        path_derivatives=["/thumbs/key.jpeg"],
    )
    db = SimpleNamespace(photos=lambda: [key])

    records = {r.uuid: r for r in iter_photo_records(db, with_derivatives=True)}
    assert set(records) == {"key-1", "sib-1"}
    assert records["sib-1"].derivative_path == "/thumbs/sib.jpeg"
    assert records["key-1"].derivative_path == "/thumbs/key.jpeg"

    plain = {r.uuid: r for r in iter_photo_records(db)}
    assert plain["sib-1"].derivative_path is None


# --- display derivative stamping (Phase 1b Task 2) ------------------------------

DISPLAY_SIZES = {
    "/thumbs/big.jpeg": (1024, 768),
    "/thumbs/small.jpeg": (480, 360),
    "/thumbs/sib-big.jpeg": (1024, 768),
    "/thumbs/sib.jpeg": (480, 360),
    "/thumbs/key.jpeg": (480, 360),
}


def test_neither_derivative_is_resolved_unless_asked():
    p = make_fake_photoinfo(
        path_derivatives=["/thumbs/big.jpeg", "/thumbs/small.jpeg"]
    )
    r = record_from_photoinfo(p)
    assert r.derivative_path is None and r.display_path is None


def test_stamps_both_derivatives_at_opposite_ends(monkeypatch):
    """The scan is the only place a photo's PhotoInfo is in hand, so both picks are
    made in the same pass. They must land in the right fields: swapping them would
    feed the pipeline 1024px rasters (corrupting every distance) while showing the
    owner 480px ones -- both plausible, both wrong, and neither visible in a test
    that only checks the fields are populated."""
    monkeypatch.setattr(
        "photocull.derivatives._measure", lambda path: DISPLAY_SIZES.get(path)
    )
    p = make_fake_photoinfo(
        path_derivatives=["/thumbs/big.jpeg", "/thumbs/small.jpeg"]
    )
    r = record_from_photoinfo(p, with_derivative=True)
    assert r.derivative_path == "/thumbs/small.jpeg"
    assert r.display_path == "/thumbs/big.jpeg"


def test_the_scan_measures_each_derivative_once_per_photo(monkeypatch):
    """~63 s of ImageIO header reads across the real library rides on this. Both
    fields come from one `derivative_paths` call, never from two selector calls."""
    calls = []

    def counting(path):
        calls.append(path)
        return DISPLAY_SIZES.get(path)

    monkeypatch.setattr("photocull.derivatives._measure", counting)
    p = make_fake_photoinfo(
        path_derivatives=["/thumbs/big.jpeg", "/thumbs/small.jpeg"]
    )
    record_from_photoinfo(p, with_derivative=True)
    assert sorted(calls) == ["/thumbs/big.jpeg", "/thumbs/small.jpeg"]


def test_a_photo_with_no_local_derivative_stamps_none_for_both():
    p = make_fake_photoinfo(path_derivatives=[])
    r = record_from_photoinfo(p, with_derivative=True)
    assert r.derivative_path is None and r.display_path is None


def test_burst_siblings_get_their_display_derivative_stamped_too(monkeypatch):
    """Same reason as `test_burst_siblings_get_their_derivative_stamped_too`, and it
    bites harder here: a burst sibling with no `display_path` is a photo the review
    UI cannot show at all, and burst siblings are the near-duplicates most likely to
    need comparing side by side."""
    monkeypatch.setattr(
        "photocull.derivatives._measure", lambda path: DISPLAY_SIZES.get(path)
    )
    sibling = make_fake_photoinfo(
        uuid="sib-1",
        burst=True,
        path_derivatives=["/thumbs/sib-big.jpeg", "/thumbs/sib.jpeg"],
    )
    key = make_fake_photoinfo(
        uuid="key-1",
        burst=True,
        burst_photos=[sibling],
        path_derivatives=["/thumbs/key.jpeg"],
    )
    db = SimpleNamespace(photos=lambda: [key])

    records = {r.uuid: r for r in iter_photo_records(db, with_derivatives=True)}
    assert records["sib-1"].derivative_path == "/thumbs/sib.jpeg"
    assert records["sib-1"].display_path == "/thumbs/sib-big.jpeg"
    assert records["key-1"].display_path == "/thumbs/key.jpeg"


def test_shared_album_assets_are_flagged():
    """`PhotoInfo.shared` was never read, so shared-album copies flowed into
    clustering indistinguishable from the user's own photos."""
    p = make_fake_photoinfo(uuid="s1", shared=True)
    assert record_from_photoinfo(p).is_shared is True


def test_photos_default_to_not_shared():
    assert record_from_photoinfo(make_fake_photoinfo(uuid="s2")).is_shared is False


# --- keeper_uuids (Phase 3 Task 2) ----------------------------------------------


class _FakeAlbum:
    def __init__(self, title, folder_names, photos):
        self.title = title
        self.folder_names = folder_names
        self.photos = photos


class _FakeDBWithAlbums:
    def __init__(self, albums):
        self.album_info = albums


def test_keeper_uuids_finds_cull_keepers_by_title_and_folder():
    photos = [SimpleNamespace(uuid="a"), SimpleNamespace(uuid="b")]
    db = _FakeDBWithAlbums(
        [
            _FakeAlbum("Keepers", ["Cull"], photos),
            _FakeAlbum("Keepers", ["SomeOtherFolder"], [SimpleNamespace(uuid="wrong")]),
        ]
    )
    assert keeper_uuids(db) == {"a", "b"}


def test_keeper_uuids_empty_when_no_matching_album():
    db = _FakeDBWithAlbums([_FakeAlbum("Candidates", ["Cull"], [SimpleNamespace(uuid="x")])])
    assert keeper_uuids(db) == set()


def test_keeper_uuids_empty_when_no_albums_at_all():
    db = _FakeDBWithAlbums([])
    assert keeper_uuids(db) == set()
