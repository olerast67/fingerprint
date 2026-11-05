import sys; sys.stdout.reconfigure(encoding="utf-8")
import json, re, time, urllib.request, urllib.parse
from pathlib import Path
OUT = Path("samples/internet"); OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "fingerprint-test/0.1"}
T = {  # slug: (title, role)
 "macro_zoom": ("Finger Print Zoom 01 (203533181).jpeg", "макро подушечки"),
 "macro_dedo": ("Dedo macro.JPG", "макро пальца"),
 "palm_left": ("Open Palm of the Left Hand, Fingers.jpg", "открытая ладонь"),
 "palm_left2": ("Left-palm.jpg", "открытая ладонь"),
 "peace_georgia": ("Georgia flag peace hand sign.jpg", "знак мира"),
 "peace_photo": ("A Peace sign photo.jpg", "знак мира"),
 "peace_worker": ("Manual worker's dirty hand showing a peace sign (50579532757).jpg", "знак мира, грязная рука"),
 "dark_maasai": ("Maasai woman hand with bracelets.JPG", "тёмная кожа"),
 "dark_washing": ("African Woman washing hands.jpg", "тёмная кожа, мытьё рук"),
 "phone_hand": ("Hand holding Smartphone.jpg", "рука со смартфоном"),
 "phone_man": ("SZ Shenzhen man in white shirt long sleeve May 2024 R12S human hand holding smartphone.jpg", "рука со смартфоном"),
 "glove_phone": ("Do Not Use Phone with Gloved Hand.jpg", "перчатка"),
 "glove_kits": ("Hand gloves and health kits.jpg", "перчатки"),
 "crowd_hands": ("209 raised hands.jpg", "много людей, поднятые руки"),
 "crowd_band": ("Hands up for the band.jpg", "концерт, слабый свет"),
 "boys_hands": ("Boys holding hands (6408245063).jpg", "несколько людей, руки"),
 "coffee_hands": ("Hands holding a cup of coffee, Quebec City.jpg", "руки с чашкой"),
 "neg_wood": ("Wood grain (2465484626).jpg", "негатив: дерево"),
 "neg_fabric": ("Sunfirm Striped Fabrics, objectnr KA 23373.14.jpg", "негатив: полосатая ткань"),
}
rows = []
for slug, (title, role) in T.items():
    q = urllib.parse.urlencode({"action":"query","titles":"File:"+title,"prop":"imageinfo","iiprop":"url|size|extmetadata","iiurlwidth":3000,"format":"json"})
    for att in range(5):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://commons.wikimedia.org/w/api.php?"+q, headers=UA)))
            ii = list(d["query"]["pages"].values())[0]["imageinfo"][0]; break
        except Exception as e:
            print("retry", slug, e); time.sleep(20)
    else: continue
    m = ii["extmetadata"]; g = lambda k: re.sub(r"<[^>]+>", "", m.get(k, {}).get("value", "?")).strip()
    url = ii.get("thumburl") or ii["url"]
    dst = OUT / (slug + ".jpg")
    for att in range(5):
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers=UA)).read()
            raw = OUT / ("_dl_" + slug); raw.write_bytes(data); break
        except Exception as e:
            print("retry dl", slug, e); time.sleep(30)
    else: continue
    import cv2, numpy as np
    img = cv2.imdecode(np.frombuffer(data, np.uint8), 1); raw.unlink()
    if img is None: print("bad image", slug); continue
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    rows.append((dst.name, role, "File:"+title, g("Artist"), g("LicenseShortName"), ii["descriptionurl"], ii["width"], img.shape[1], img.shape[0]))
    print(slug, img.shape, g("LicenseShortName")); time.sleep(6)
with open(OUT/"SOURCES.md", "w", encoding="utf-8") as f:
    f.write("# Источники фото (Wikimedia Commons)\n\nФайлы скачаны через Commons API, уменьшены до ширины не более 3000 px и пересохранены как JPEG.\n\n")
    f.write("| Файл | Что на фото | Название на Commons | Автор | Лицензия | Страница | Оригинал px | Здесь px |\n|---|---|---|---|---|---|---|---|\n")
    for r in rows: f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3][:60]} | {r[4]} | {r[5]} | {r[6]} | {r[7]}x{r[8]} |\n")
