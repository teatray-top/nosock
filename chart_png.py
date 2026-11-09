import base64
import io
import sys

from PIL import Image, ImageDraw

MIN_SIDE = 280


def render(bars, prev_close=None, width=480, height=320, ma_n=20, show_last=12, title="", interval="1m", tz_label="ET"):
    if width < MIN_SIDE or height < MIN_SIDE:
        raise ValueError(f"image must be at least {MIN_SIDE}px per side")
    xs = [b[1] for b in bars]
    n = len(xs)
    if n == 0:
        raise ValueError("no bars")
    span = f"{bars[0][0]}-{bars[-1][0]} {tz_label}"
    closes = [b[3] for b in xs]
    ma = [sum(closes[i - ma_n + 1:i + 1]) / ma_n if i >= ma_n - 1 else None for i in range(n)]
    cum_pv = cum_v = 0.0
    vwap = []
    for b in xs:
        tp = (b[1] + b[2] + b[3]) / 3
        cum_pv += tp * b[4]
        cum_v += b[4]
        vwap.append(cum_pv / cum_v if cum_v else None)
    k0 = max(0, n - show_last)
    seg = xs[k0:]
    seg_ma = ma[k0:]
    seg_vw = vwap[k0:]
    m = len(seg)
    vals = [b[1] for b in seg] + [b[2] for b in seg] + [v for v in seg_ma if v is not None] + [v for v in seg_vw if v is not None]
    if prev_close:
        vals.append(prev_close)
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 0.01
    lo, hi = lo - pad, hi + pad
    top, bottom, left, right = 46, height - 60, 8, width - 64
    vol_top, vol_bottom = height - 54, height - 8
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)

    def y(p):
        return bottom - (p - lo) / (hi - lo) * (bottom - top)

    step = (right - left) / m
    vmax = max(b[4] for b in seg) or 1
    for i, (o, h, l, c, v) in enumerate(seg):
        cx = left + step * (i + 0.5)
        col = (200, 30, 30) if c >= o else (30, 90, 200)
        d.line([(cx, y(h)), (cx, y(l))], fill=col, width=1)
        x0, x1 = cx - step * 0.3, cx + step * 0.3
        y0, y1 = y(max(o, c)), y(min(o, c))
        d.rectangle([x0, y0, x1, max(y1, y0 + 1)], fill=col, outline=col)
        vh = (v / vmax) * (vol_bottom - vol_top)
        d.rectangle([x0, vol_bottom - vh, x1, vol_bottom], fill=(170, 170, 170))
    pts_ma = [(left + step * (i + 0.5), y(v)) for i, v in enumerate(seg_ma) if v is not None]
    pts_vw = [(left + step * (i + 0.5), y(v)) for i, v in enumerate(seg_vw) if v is not None]
    drawn = []
    if len(pts_ma) > 1:
        d.line(pts_ma, fill=(255, 140, 0), width=2)
        drawn.append(f"orange MA{ma_n}({interval} closes)")
    if len(pts_vw) > 1:
        d.line(pts_vw, fill=(120, 0, 160), width=2)
        drawn.append(f"purple VWAP = cum. over all {n} bars {span}")
    if prev_close:
        d.line([(left, y(prev_close)), (right, y(prev_close))], fill=(0, 0, 0), width=1)
        d.text((right + 3, y(prev_close) - 6), f"PC {prev_close:.2f}", fill=(0, 0, 0))
    d.text((right + 3, y(closes[-1]) - 6), f"C {closes[-1]:.2f}", fill=(200, 30, 30))
    d.text((right + 3, top - 2), f"H {hi - pad:.2f}", fill=(90, 90, 90))
    d.text((right + 3, bottom - 10), f"L {lo + pad:.2f}", fill=(90, 90, 90))
    lines = [f"{title} last {m} of {n} {interval} candles ({bars[k0][0]}-{bars[-1][0]} {tz_label}); red up, blue down", "; ".join(drawn), "black line = prev close; grey bars = volume" if prev_close else "grey bars = volume"]
    for i, s in enumerate(lines):
        d.text((left, 3 + 11 * i), s, fill=(0, 0, 0))
    return img, dict(bars_total=n, bars_drawn=m, ma_points=len(pts_ma), vwap_points=len(pts_vw), interval=interval, input_span=span, drawn_span=f"{bars[k0][0]}-{bars[-1][0]} {tz_label}", label=lines)


def to_base64_png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), len(buf.getvalue())


if __name__ == "__main__":
    import random
    random.seed(1)
    p = 100.0
    bars = []
    for i in range(30):
        o = p
        c = o * (1 + random.uniform(-0.01, 0.01))
        h, l = max(o, c) * (1 + random.uniform(0, 0.004)), min(o, c) * (1 - random.uniform(0, 0.004))
        bars.append((i, [round(o, 2), round(h, 2), round(l, 2), round(c, 2), random.randint(1000, 9000)]))
        p = c
    img, info = render(bars, prev_close=99.5, title="TEST")
    b64, nbytes = to_base64_png(img)
    out = sys.argv[1] if len(sys.argv) > 1 else "runs/chart_png_selftest.png"
    img.save(out)
    print("saved", out, img.size, "png bytes", nbytes, "base64 chars", len(b64), info)
    img2, info2 = render(bars[:8], prev_close=99.5, title="SHORT")
    print("short history:", img2.size, info2)
