"""
Test lỗi đua ghi checkpoint khi nhiều tiến trình (mỗi GPU một tiến trình) cùng ghi một file, và cache kiểm kê.

Lỗi gốc (lần chạy full VJOL 2026-10-28): write_json_atomic dùng tên tạm cố định <sha>.json.tmp, hai tiến trình
ghi cùng lúc thì một bên mất file tạm, os.replace ném FileNotFoundError và tiến trình dừng.
"""

import multiprocessing as mp
from pathlib import Path

from vi_corpus.common.batch_extract import file_sha256, inventory
from vi_corpus.common.state import read_json, write_json_atomic


def _hammer(path: str, n: int, tag: int, errors) -> None:
    """Ghi n lần vào cùng một file; báo lỗi (nếu có) qua hàng đợi `errors`."""
    try:
        for i in range(n):
            write_json_atomic(Path(path), {"tag": tag, "i": i, "pad": "x" * 2000})
    except Exception as exc:  # noqa: BLE001
        errors.put(repr(exc))


def test_concurrent_writers_do_not_collide(tmp_path):
    """8 tiến trình ghi dồn dập cùng một checkpoint: không lỗi, file cuối đọc được, không sót file tạm."""
    target = tmp_path / "abc.json"
    errors = mp.Queue()
    procs = [mp.Process(target=_hammer, args=(str(target), 300, t, errors)) for t in range(8)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert errors.empty(), errors.get()
    assert read_json(target)["pad"] == "x" * 2000
    assert [p.name for p in tmp_path.iterdir()] == ["abc.json"]


def test_failed_write_leaves_no_tmp(tmp_path):
    """Dữ liệu không ghi được JSON: ném lỗi và không để file tạm mồ côi."""
    try:
        write_json_atomic(tmp_path / "a.json", {"bad": object()})
    except TypeError:
        pass
    assert list(tmp_path.iterdir()) == []


def test_inventory_cache_skips_unchanged_and_rehashes_changed(tmp_path, monkeypatch):
    """Lần 2 không băm lại file chưa đổi; file đổi nội dung (size/mtime) thì băm lại; kết quả giống nhau."""
    (tmp_path / "a.pdf").write_bytes(b"one")
    (tmp_path / "b.pdf").write_bytes(b"one")  # trùng nội dung với a
    (tmp_path / "c.pdf").write_bytes(b"two")
    cache = tmp_path / "cache.json"
    first = inventory(tmp_path, files=sorted(tmp_path.glob("*.pdf")), cache_path=cache)
    assert sorted(len(v) for v in first.values()) == [1, 2]

    calls = []
    import vi_corpus.common.batch_extract as be
    real = be.file_sha256
    monkeypatch.setattr(be, "file_sha256", lambda p: calls.append(p) or real(p))
    again = inventory(tmp_path, files=sorted(tmp_path.glob("*.pdf")), cache_path=cache)
    assert again == first and calls == []

    (tmp_path / "c.pdf").write_bytes(b"three!")
    changed = inventory(tmp_path, files=sorted(tmp_path.glob("*.pdf")), cache_path=cache)
    assert calls == [tmp_path / "c.pdf"]
    assert real(tmp_path / "c.pdf") in changed
