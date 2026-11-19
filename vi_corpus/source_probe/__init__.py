"""
Kiểm tra nhanh chất lượng các nguồn ứng viên (docs/SOURCE_CANDIDATES.md) trước khi chốt: lấy mẫu nhỏ từ xa (không tải cả bộ),
đo bằng chính các số đo của pipeline (language, quality) cộng vài số đo riêng cho dữ liệu dịch máy / có công thức.

- specs.py:   danh sách nguồn cần thử và cách đọc từng nguồn.
- sample.py:  lấy mẫu từ xa (parquet trên Hugging Face theo row group, API datasets-server, shard jsonl.zst của HPLT).
- measure.py: đo từng bản ghi mẫu và tổng hợp theo nguồn.
"""
