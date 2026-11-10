"""
Lọc câu "nói về chính ảnh" của OCR VLM lọt vào văn bản (J-33 C): khi trang không có chữ hoặc mô hình không đọc được, Qwen3-VL trả một câu
tiếng Anh thay cho nội dung ("There is no text in the image.", "The image is blank.", "I cannot read the text ..."). Câu này không phải
nội dung bài báo; ở vjol_10k_v5 nó lọt vào 4 đoạn band A (đầu hoặc giữa đoạn) và 5 đoạn khác chỉ bị loại vì quá ngắn.

Bộ kiểm trang của OCR (vi_corpus.ocr_bakeoff.guard) chỉ bắt đầu ra MỞ ĐẦU bằng lời suy nghĩ ("The user wants me to ..."), không bắt các câu này,
nên lọc ở đây: mỗi câu (hoặc dòng) khớp mẫu bị bỏ, phần còn lại của trang giữ nguyên. Mẫu viết chặt (cả câu, cụm từ cố định) để câu văn bình thường
nhắc tới "text" hay "image" không bị mất. Dùng ở bước trích (vi_corpus.common.pdf_text.extract_pdf: trang chỉ còn câu này thành trang rỗng "blank")
và lúc làm sạch bài (vi_corpus.vjol.clean) để checkpoint cũ cũng được lọc.
"""

import re

# Cụm câu OCR VLM hay trả khi trang trống / không đọc được. Mỗi cụm đều nhắc tới CHÍNH ẢNH ("in the image", "this page image"), nên câu văn
# thường nhắc tới "no text" hay "the document" không bị khớp. Không phân biệt hoa thường; dấu chấm cuối câu đi kèm.
_IMAGE_REF = r"(?:the|this|that) (?:page |scanned )?(?:image|picture|scan|photo|page)"
_STRICT = (
    rf"there (?:is|are) no (?:visible |readable |legible |discernible )?(?:text|content|writing)(?: (?:visible|present|found|detected))? (?:in|on|within) {_IMAGE_REF}",
    rf"{_IMAGE_REF} (?:is|appears to be|seems to be) (?:completely |entirely |totally )?(?:blank|empty|unreadable|illegible)",
    rf"{_IMAGE_REF} (?:does not|doesn't) (?:contain|have|show|include) (?:any )?(?:visible |readable )?(?:text|content|writing)",
    rf"no (?:visible |readable |legible )?text (?:is |was )?(?:found|visible|present|detected) (?:in|on) {_IMAGE_REF}",
    rf"i (?:cannot|can't|am unable to|could not) (?:read|see|transcribe|extract) (?:any |the )?(?:visible |readable )?(?:text|content|writing)(?: from| in| on)? {_IMAGE_REF}",
)
_SENTENCE = re.compile(r"\b(?:" + "|".join(f"(?:{s})" for s in _STRICT) + r")\s*[.!]?", re.IGNORECASE)
# Dòng chỉ gồm một nhãn "trống" (không cần nhắc tới ảnh): bỏ nguyên dòng, vì trong văn bản đã làm sạch một dòng như vậy không phải nội dung.
_WHOLE_LINE = re.compile(r"^\W*(?:there is no text|no text|blank page|empty page|blank|empty)\W*$", re.IGNORECASE)


def strip_vlm_noise(text: str) -> tuple[str, int]:
    """
    Bỏ câu "nói về chính ảnh" của OCR VLM khỏi văn bản một trang.

    Dòng chỉ gồm câu như vậy bị bỏ nguyên dòng; câu dính đầu hoặc giữa dòng ("There is no text in the image. Chương 1 ...") bị cắt khỏi dòng, phần còn lại giữ nguyên.

    Returns:
        (văn bản đã lọc, số câu đã bỏ).
    """
    removed = 0
    kept: list[str] = []
    for line in text.split("\n"):
        if _WHOLE_LINE.match(line):
            removed += 1
            continue
        new, n = _SENTENCE.subn(" ", line)
        if n:
            removed += n
            line = re.sub(r"[ \t]{2,}", " ", new).strip()
            if not re.search(r"\w", line):
                continue
        kept.append(line)
    return "\n".join(kept), removed
