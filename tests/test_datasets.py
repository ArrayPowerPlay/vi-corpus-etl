"""
Test cho sổ đăng ký bộ dữ liệu (vi_corpus.download.datasets) và việc phân loại lỗi retry (vi_corpus.download.hub).

Không cần mạng.
"""

import pytest

from vi_corpus.download.datasets import DATASETS, get_spec
from vi_corpus.download.hub import CorruptDownloadError, _is_retryable


def test_three_datasets_in_order():
    """Đủ 3 bộ, bộ nhỏ (Instruct) chạy trước."""
    assert list(DATASETS) == ["sea_instruct_2602", "sea_pile_v2", "sea_lion_pile_v1"]


def test_vietnamese_dirs():
    """Mỗi bộ trỏ đúng thư mục tiếng Việt trên HF."""
    assert get_spec("sea_pile_v2").vi_dir == "vi"
    assert get_spec("sea_lion_pile_v1").vi_dir == "sea-pile-mc4/vi"
    assert get_spec("sea_instruct_2602").vi_dir == "Vietnamese"


def test_unknown_key_raises():
    """Mã sai thì báo lỗi kèm danh sách mã hợp lệ."""
    with pytest.raises(KeyError, match="sea_pile_v2"):
        get_spec("khong_ton_tai")


def test_retryable_classification():
    """Lỗi mạng / file hỏng thì thử lại."""
    assert _is_retryable(ConnectionError("mất mạng"))
    assert _is_retryable(CorruptDownloadError("sai kích thước"))
