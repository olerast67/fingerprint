import sys; sys.stdout.reconfigure(encoding="utf-8")
import json, urllib.request, urllib.parse, re
sys_ok=1
q = sys.argv[1]; n = sys.argv[2] if len(sys.argv)>2 else "12"
u = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({"action":"query","generator":"search","gsrsearch":q+" filetype:bitmap","gsrnamespace":6,"gsrlimit":n,"prop":"imageinfo","iiprop":"url|size|extmetadata|mime","iiurlwidth":4000,"format":"json"})
d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent":"fingerprint-test/0.1"})))
for p in sorted(d.get("query",{}).get("pages",{}).values(), key=lambda p:p.get("index",0)):
    ii = p["imageinfo"][0]; m = ii["extmetadata"]
    print(p["title"], ii["width"], ii["height"], ii["mime"], re.sub("<[^>]+>","",m.get("LicenseShortName",{}).get("value","?")))
