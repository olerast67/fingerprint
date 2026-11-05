import sys; sys.stdout.reconfigure(encoding="utf-8")
import json, urllib.request, urllib.parse, re, time
for q in sys.argv[1:]:
    u = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({"action":"query","generator":"search","gsrsearch":q+" filetype:bitmap","gsrnamespace":6,"gsrlimit":8,"prop":"imageinfo","iiprop":"size|extmetadata|mime","format":"json"})
    for a in range(4):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent":"fingerprint-test/0.1"}))); break
        except Exception as e: time.sleep(15); d={}
    print("##", q)
    for p in sorted(d.get("query",{}).get("pages",{}).values(), key=lambda p:p.get("index",0)):
        ii = p["imageinfo"][0]; m = ii["extmetadata"]
        lic = re.sub("<[^>]+>","",m.get("LicenseShortName",{}).get("value","?"))
        if ii["width"]>=1200 and ii["mime"]=="image/jpeg": print(p["title"], ii["width"], ii["height"], lic)
    time.sleep(3)
