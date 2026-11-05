"""Обёртка над dev/js_check.py: приводит маску MediaPipe к форме (256,256), остальное без изменений."""
import sys
from pathlib import Path
p = Path(__file__).resolve().parent.parent / "js_check.py"
src = p.read_text(encoding="utf-8").replace("confidence_masks[i].numpy_view()", "confidence_masks[i].numpy_view().reshape(256, 256)")
sys.path.insert(0, str(p.parent))
exec(compile(src, str(p), "exec"), {"__file__": str(p), "__name__": "__main__"})
