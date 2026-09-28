"""
vi_corpus — pipeline ETL xây corpus tiếng Việt cho LLM từ nhiều nguồn (SEA, Drive, VISTA/VJOL).

Các gói con:
    download : tải phần tiếng Việt của dataset SEA từ Hugging Face (có checkpoint từng file).
    schema   : schema chung của clean corpus (CORPUS_SCHEMA, tính KPI truy vết nguồn).
    sources  : sổ đăng ký nguồn (owner, license, domain, đường dẫn) và adapter đọc nguồn text.
Các gói xử lý còn lại (stages, viz) sẽ được bổ sung theo docs/ROADMAP.md.
"""
