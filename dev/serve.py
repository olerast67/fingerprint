"""Локальный сервер для проверки страницы: только 127.0.0.1, правильные типы файлов.

Стандартный http.server на Windows берёт типы из реестра и отдаёт .js как text/plain,
браузер такие модули не запускает. Запуск: python dev/serve.py [порт]
"""
import functools
import http.server
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm",
        ".css": "text/css", ".html": "text/html; charset=utf-8", ".tflite": "application/octet-stream",
    }

    def do_POST(self):
        # только для проверки: страница отдаёт сюда готовый файл, он ложится в samples/private/out/browser
        from urllib.parse import parse_qs, urlparse
        u = urlparse(self.path)
        name = Path(parse_qs(u.query).get("name", ["out.jpg"])[0]).name
        if u.path != "/__save" or self.client_address[0] != "127.0.0.1":
            self.send_error(404)
            return
        out = ROOT / "samples" / "private" / "out" / "browser"
        out.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(204)
        self.end_headers()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")   # без кэша: после правки браузер берёт новый файл
        super().end_headers()


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=str(ROOT))).serve_forever()
