import sys, json, re
for f in sys.argv[1:]:
    for l in open(f, encoding="utf-8", errors="replace"):
        m = re.match(r"^(\S+\.jpg) (\{.*\})\s*$", l)
        if not m: continue
        s = json.loads(m.group(2))
        print(f"{m.group(1)[:-4]}\tP={s['P']:.0f}\tmask={s['maskFrac']:.3f}\tfar={s['farFrac']:.3f}\thits={s['hits']}\tres={s.get('residual')}\tms={s['ms']}")
