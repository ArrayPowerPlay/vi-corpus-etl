"""
Test adapter NeMo Curator + Ray (vi_corpus/pipeline/curator.py) trên CPU với Ray cục bộ.

Bỏ qua nếu chưa cài nemo-curator (`uv sync --group curator`). Kiểm tra: kết quả chạy song song giống hệt chạy tuần tự,
chạy được hai stage liên tiếp trong cùng một backend (executor tắt Ray sau mỗi lần chạy), và kiểu dữ liệu không bị đổi.
"""

import os

import pytest

pytest.importorskip("nemo_curator")
pytest.importorskip("ray")
os.environ.setdefault("RAY_AUTH_MODE", "disabled")

from tests.test_pipeline import make_cfg, vi_text  # noqa: E402
from vi_corpus.pipeline.curator import CuratorBackend, decode_rows, encode_rows  # noqa: E402
from vi_corpus.pipeline.language import annotate_language  # noqa: E402
from vi_corpus.pipeline.quality import annotate_quality  # noqa: E402


def _rows(n: int) -> list[dict]:
    """Bản ghi giả đủ cột cho stage language / quality."""
    return [{"doc_id": f"d{i}", "source_key": "sea_pile_v2", "text": vi_text(25, i), "word_count": 100, "n": i,
             "maybe": None} for i in range(n)]


def test_encode_decode_giu_nguyen_kieu():
    """encode/decode không đổi int thành float, giữ None và bytes (doc_minhash)."""
    rows = _rows(3)
    rows[0]["doc_minhash"] = b"\x00\x01\xff"
    assert decode_rows(encode_rows(rows)) == rows


@pytest.mark.parametrize("executor", ["ray_actor_pool", "xenna"])
def test_song_song_giong_tuan_tu(executor):
    """Hai stage liên tiếp qua executor cho kết quả y hệt hàm tuần tự, đúng thứ tự đầu vào."""
    cfg = make_cfg(mix={"sea_pile_v2": 30})
    expected = annotate_quality(annotate_language(_rows(30), cfg.lang_model), cfg)
    with CuratorBackend(executor, num_cpus=2, num_gpus=0, rows_per_task=7) as backend:
        got = backend.quality(backend.language(_rows(30), cfg), cfg)
    assert got == expected
