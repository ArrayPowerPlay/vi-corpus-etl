"""
Pipeline xử lý chung (ingest -> normalize -> chunk -> language -> quality -> dedup -> knowledge unit -> audit -> report).

Mỗi stage đọc Parquet của stage trước và ghi Parquet mới (xem docs/PIPELINE.md); runner.py nối các stage lại,
có checkpoint theo stage. Các stage là hàm thuần trên list[dict] nên test được mà không cần dữ liệu thật.
"""
