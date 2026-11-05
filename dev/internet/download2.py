import sys; sys.stdout.reconfigure(encoding="utf-8")
import json, re, time, urllib.request, urllib.parse
from pathlib import Path
import cv2, numpy as np
OUT = Path("samples/diversity"); OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "fingerprint-test/0.1"}
titles = [l.strip() for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
meta = OUT/"meta.jsonl"
done = set()
if meta.exists(): done = {json.loads(l)["title"] for l in meta.read_text(encoding="utf-8").splitlines() if l}
for title in titles:
    if title in done: continue
    q = urllib.parse.urlencode({"action":"query","titles":"File:"+title,"prop":"imageinfo","iiprop":"url|size|extmetadata","iiurlwidth":1920,"format":"json"})
    ii=None
    for att in range(4):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://commons.wikimedia.org/w/api.php?"+q, headers=UA)))
            ii = list(d["query"]["pages"].values())[0]["imageinfo"][0]; break
        except Exception as e: print("retry", title, e); time.sleep(15)
    if not ii: continue
    m = ii["extmetadata"]; g = lambda k: re.sub(r"<[^>]+>", "", m.get(k, {}).get("value", "?")).strip()
    data=None
    for att in range(4):
        try: data = urllib.request.urlopen(urllib.request.Request(ii.get("thumburl") or ii["url"], headers=UA)).read(); break
        except Exception as e: print("retry dl", title, e); time.sleep(20)
    if not data: continue
    img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
    if img is None: print("bad", title); continue
    slug = re.sub(r"[^A-Za-z0-9]+","_",title.rsplit(".",1)[0])[:40].strip("_")
    cv2.imwrite(str(OUT/(slug+".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(meta,"a",encoding="utf-8") as f:
        f.write(json.dumps({"title":title,"slug":slug,"author":g("Artist")[:70],"license":g("LicenseShortName"),"url":ii["descriptionurl"],"orig":[ii["width"],ii["height"]],"size":[img.shape[1],img.shape[0]]},ensure_ascii=False)+"\n")
    print(slug, img.shape[1], img.shape[0], g("LicenseShortName"), flush=True); time.sleep(4)
