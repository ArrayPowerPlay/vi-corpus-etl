# Makefile: các lệnh gọn cho việc hay chạy trên server.
#
# make count-ocr ROOT=<data-root> [JOBS=16]
#   Đếm CHÍNH XÁC số trang cần OCR của VJOL và VISTA, chỉ dùng CPU (không GPU):
#   1) trích --no-ocr (JOBS tiến trình) để mọi PDF có checkpoint bản hiện hành (file đã xong thì bỏ qua),
#   2) đếm lại, in tóm tắt và ghi <ROOT>/processed/ocr_count/{vjol,vista}.json + *_by_year.csv.
#   Chỉ tin con số khi "CHƯA quét", "bản trích cũ" và "lỗi" đều bằng 0.
#   KHÔNG chạy khi lượt GPU của VJOL / VISTA đang chạy (hai bên tranh file).

ROOT ?= $(SEA_DATA_ROOT)
JOBS ?= 16
SOURCES ?= vjol vista
PY := uv run --no-sync python

.PHONY: count-ocr
count-ocr:
	@test -n "$(ROOT)" || { echo "Thiếu ROOT: make count-ocr ROOT=<data-root>"; exit 1; }
	@mkdir -p "$(ROOT)/logs"
	@for s in $(SOURCES); do \
	  echo "== Trích --no-ocr $$s ($(JOBS) tiến trình) =="; \
	  for i in $$(seq 1 $(JOBS)); do \
	    $(PY) scripts/$$s/extract.py --data-root "$(ROOT)" --no-ocr --no-export --worker-name cnt$$i \
	      > "$(ROOT)/logs/$${s}_count_cpu$$i.log" 2>&1 & \
	  done; wait; \
	done
	$(PY) scripts/count_ocr_pages.py --data-root "$(ROOT)" --sources $(shell echo $(SOURCES) | tr ' ' ',')
