import sys; sys.stdout.reconfigure(encoding="utf-8")
import json, urllib.request, urllib.parse, re, time
for c in sys.argv[1:]:
    u = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({"action":"query","generator":"categorymembers","gcmtitle":"Category:"+c,"gcmtype":"file","gcmlimit":60,"prop":"imageinfo","iiprop":"size|extmetadata|mime","format":"json"})
    d={}
    for a in range(4):
        try: d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent":"fingerprint-test/0.1"}))); break
        except Exception as e: time.sleep(15)
    print("##", c)
    for p in d.get("query",{}).get("pages",{}).values():
        ii = p["imageinfo"][0]; m = ii["extmetadata"]
        lic = re.sub("<[^>]+>","",m.get("LicenseShortName",{}).get("value","?"))
        if ii["width"]>=1500 and ii["mime"]=="image/jpeg" and ("CC" in lic or "ublic" in lic): print(p["title"][5:], ii["width"], lic)
    time.sleep(3)
