"""
Test cho vi_corpus.download.checkpoint và logic "file đã xong chưa" trong downloader.

Không cần mạng: chỉ làm việc với file tạm trên đĩa.
"""

import json

import pytest

from vi_corpus.download.checkpoint import Checkpoint, write_json_atomic
from vi_corpus.download.downloader import _is_complete, acquire_run_lock, cleanup_incomplete
from vi_corpus.download.hub import RemoteFile


def test_write_json_atomic_leaves_no_tmp(tmp_path):
    """Ghi xong thì chỉ còn file thật, không còn file .tmp."""
    target = tmp_path / "a" / "state.json"
    write_json_atomic(target, {"x": 1})
    assert json.loads(target.read_text()) == {"x": 1}
    assert list(target.parent.iterdir()) == [target]


def test_state_path_is_flat(tmp_path):
    """Đường dẫn có "/" được làm phẳng thành một file trong thư mục state."""
    ckpt = Checkpoint(tmp_path)
    p = ckpt.state_path("sea-pile-mc4/vi/mc4-vi-00000-00328.jsonl.gz")
    assert p.parent == tmp_path
    assert p.name == "sea-pile-mc4__vi__mc4-vi-00000-00328.jsonl.gz.json"


def test_is_done_checks_size_and_sha(tmp_path):
    """Chỉ coi là xong khi state 'done' và kích thước/sha256 khớp với HF."""
    ckpt = Checkpoint(tmp_path)
    assert not ckpt.is_done("vi/a.parquet", 10, "abc")

    ckpt.mark_done("vi/a.parquet", {"size": 10, "sha256": "abc"})
    assert ckpt.is_done("vi/a.parquet", 10, "abc")
    assert not ckpt.is_done("vi/a.parquet", 11, "abc")  # tác giả cập nhật file
    assert not ckpt.is_done("vi/a.parquet", 10, "def")


def test_failed_is_not_done_and_is_listed(tmp_path):
    """File lỗi không được coi là xong, và xuất hiện trong all_states()."""
    ckpt = Checkpoint(tmp_path)
    ckpt.mark_failed("vi/b.parquet", "timeout")
    assert not ckpt.is_done("vi/b.parquet", 10, None)
    assert [s["status"] for s in ckpt.all_states()] == ["failed"]


def test_manifest_not_in_states(tmp_path):
    """Manifest không bị tính là state của một file."""
    ckpt = Checkpoint(tmp_path)
    ckpt.write_manifest({"files": []})
    assert ckpt.read_manifest() == {"files": []}
    assert ckpt.all_states() == []


def test_is_complete_requires_file_on_disk(tmp_path):
    """State 'done' nhưng file trên đĩa bị xoá/thiếu thì phải tải lại."""
    raw, state = tmp_path / "raw", tmp_path / "state"
    ckpt = Checkpoint(state)
    remote = RemoteFile(path="vi/c.parquet", size=4, sha256=None)
    ckpt.mark_done(remote.path, {"size": 4, "sha256": None})

    assert not _is_complete(ckpt, raw, remote)  # chưa có file
    (raw / "vi").mkdir(parents=True)
    (raw / "vi" / "c.parquet").write_bytes(b"12")
    assert not _is_complete(ckpt, raw, remote)  # file thiếu dữ liệu
    (raw / "vi" / "c.parquet").write_bytes(b"1234")
    assert _is_complete(ckpt, raw, remote)


def test_cleanup_incomplete_removes_leftovers(tmp_path):
    """File tải dở (*.incomplete) sau crash được dọn, file đã tải xong thì giữ nguyên."""
    cache = tmp_path / ".cache" / "huggingface" / "download" / "vi"
    cache.mkdir(parents=True)
    (cache / "abc.123.incomplete").write_bytes(b"x" * 10)
    (tmp_path / "vi").mkdir()
    (tmp_path / "vi" / "a.parquet").write_bytes(b"ok")

    cleanup_incomplete(tmp_path)
    assert not list(cache.glob("*.incomplete"))
    assert (tmp_path / "vi" / "a.parquet").exists()


def test_run_lock_blocks_second_holder(tmp_path):
    """Khi đã có người giữ khoá thì lần xin khoá thứ hai phải báo lỗi."""
    first = acquire_run_lock(tmp_path)
    with pytest.raises(SystemExit, match="tiến trình khác"):
        acquire_run_lock(tmp_path)
    first.close()  # nhả khoá
    acquire_run_lock(tmp_path).close()
