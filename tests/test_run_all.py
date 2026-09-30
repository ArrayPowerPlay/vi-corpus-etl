"""Test cho scripts/run_all.py: bỏ qua phần thiếu raw, phần lỗi không chặn phần sau, truyền --data-root/--status."""

import sys

from scripts.run_all import run_parts


def test_bo_qua_loi_va_truyen_tham_so(tmp_path, capsys):
    """Phần thiếu raw bị bỏ qua; phần lỗi (mã 3) không chặn phần sau; script con nhận đúng tham số."""
    (tmp_path / "raw/b").mkdir(parents=True)
    echo = tmp_path / "echo.py"
    echo.write_text("import sys; print('ARGS', *sys.argv[1:]); sys.exit(3 if 'fail' in sys.argv[0] else 0)")
    fail = tmp_path / "fail.py"
    fail.write_text("import sys; sys.exit(3)")
    parts = {"a": ("raw/a", str(echo)), "b": ("raw/b", str(fail)), "c": (None, str(echo))}

    res = run_parts(["a", "b", "c"], tmp_path, status=True, parts=parts)

    assert res == {"a": "bỏ qua", "b": "lỗi (mã 3)", "c": "ok"}
    out = capsys.readouterr().out
    assert "Bỏ qua a" in out
    assert out.count(f"--data-root {tmp_path} --status") >= 2  # dòng lệnh của b và c
