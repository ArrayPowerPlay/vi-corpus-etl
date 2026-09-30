"""
vi_corpus.sea — tải phần tiếng Việt của các bộ dữ liệu SEA (AI Singapore) từ Hugging Face.

Các module:
    datasets    : "sổ đăng ký" các bộ dữ liệu (repo, thư mục tiếng Việt).
    hub         : làm việc với Hugging Face (token, liệt kê file, tải có retry, kiểm tra sha256).
    checkpoint  : lưu / đọc tiến trình từng file để chạy tiếp sau khi crash.
    downloader  : vòng tải song song cho một bộ dữ liệu + xem tiến độ.
    cli         : tham số dòng lệnh dùng chung cho các script trong thư mục scripts/.
"""
