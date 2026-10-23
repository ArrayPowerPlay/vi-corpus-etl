"""
Nhận diện mục tài liệu tham khảo theo MẪU ĐẦU DÒNG (J-25 E), dùng chung cho vi_corpus.vjol.clean (cắt phần tham khảo) và stage quality (số đo ref_line, nhãn reference_list).

Cách cũ coi dòng là mục tham khảo khi "có năm và có một từ khoá (Tạp chí, tr., et al ...) ở đâu cũng được", nên thơ / văn xuôi có trích dẫn kiểu "(Họ, 1997, tr.623)" bị gắn nhãn nhầm
(126 đoạn gắn nhãn, chỉ ~18 giống danh mục thật ở vjol_10k_v3). Nay một dòng là mục tham khảo khi đồng thời (1) BẮT ĐẦU bằng mẫu tham khảo, (2) có năm xuất bản và (3) có một từ khoá thư mục:
- "[12] ...", "12. Nguyễn Văn A ...", "12) ..." (số thứ tự, sau đó là tên tác giả viết hoa),
- "Họ, X. ..." (Smith, J. K.), "Nguyen V.A., Tran T.B. ...", "Nguyễn Văn A (2010) ...", "Trần B, Lê C (2004) ...".
Mục dài thường xuống dòng nên reference_share tính cả tối đa MAX_CONTINUATION dòng nối tiếp sau một mục.
"""

import re

_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
# từ khoá thư mục: chữ cái đầu tên ("Nguyen V.A."), et al, nhà xuất bản, tạp chí, trang, DOI, liên kết
_CUE = re.compile(r"\b[A-ZĐ]\.\s?[A-ZĐ]?\.?|et al|\bNXB\b|\bNxb\b|Nhà xuất bản|Tạp chí|Journal|Press\b|\bpp?\.|\btr\.|\bdoi\b|https?://|\bVol\b|\bNo\b\.?")
_NAME = r"[A-ZĐ][^\W\d_]*"  # một từ viết hoa (họ, tên đệm, tên)
_STARTS = (
    re.compile(rf"^\s*\[\d{{1,3}}\]\s*{_NAME}"),  # [12] Nguyễn ...
    re.compile(rf"^\s*\d{{1,3}}[.)]\s+{_NAME}"),  # 12. Nguyễn ... / 12) Smith ...
    re.compile(rf"^\s*{_NAME}(?:[-']{_NAME})?,\s*[A-ZĐ]\."),  # Smith, J.
    re.compile(rf"^\s*{_NAME}\s+[A-ZĐ]\.\s?(?:[A-ZĐ]\.)?"),  # Nguyen V.A., Tran T.B. (chữ cái đầu tên đứng sau họ, không dấu phẩy)
    re.compile(rf"^\s*(?:{_NAME}\.?\s+){{1,4}}{_NAME}\s*[,(]"),  # Nguyễn Văn A (2010) / Trần B, Lê C (2004)
)
MAX_CONTINUATION = 3  # số dòng nối tiếp tối đa được tính vào một mục tham khảo


def is_reference_entry(line: str) -> bool:
    """True nếu dòng là DÒNG ĐẦU của một mục tài liệu tham khảo: bắt đầu bằng mẫu tham khảo và có năm xuất bản và một từ khoá thư mục."""
    return bool(_YEAR.search(line) and _CUE.search(line) and any(p.match(line) for p in _STARTS))


def reference_flags(lines: list[str]) -> list[bool]:
    """
    Cờ "thuộc phần tham khảo" cho từng dòng: dòng đầu mục (is_reference_entry) và tối đa MAX_CONTINUATION dòng nối tiếp ngắn (<= 30 từ, không phải đầu mục khác)
    ngay sau nó.
    """
    flags: list[bool] = []
    run = 0  # số dòng nối tiếp còn được phép
    for ln in lines:
        if is_reference_entry(ln):
            flags.append(True)
            run = MAX_CONTINUATION
        elif run > 0 and len(ln.split()) <= 30:
            flags.append(True)
            run -= 1
        else:
            flags.append(False)
            run = 0
    return flags


def reference_share(lines: list[str]) -> float:
    """Tỷ lệ dòng thuộc phần tham khảo (reference_flags) trên các dòng đã cho; 0.0 khi không có dòng."""
    return sum(reference_flags(lines)) / len(lines) if lines else 0.0
