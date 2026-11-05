"""Render a single-scene black-background SpaceFL concept animation."""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 675
FRAMES, DELAY = 72, 90
OUT = Path(__file__).resolve().parents[2] / "output" / "spacefl_minimal_black.gif"
FONT = "C:/Windows/Fonts/msyh.ttc"


def font(size: int, bold: bool = False):
    path = FONT if Path(FONT).exists() else "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"
    return ImageFont.truetype(path, size)


F12, F14, F28 = font(12), font(14), font(28, True)
N_SATS = 6
STATION_ANGLE = -0.58


def draw_label(draw, x, y, text, color, fnt=F12):
    box = draw.textbbox((0, 0), text, font=fnt)
    width = box[2] - box[0] + 18
    draw.rounded_rectangle((x, y, x + width, y + 25), radius=2, fill=(11, 11, 11), outline=color, width=1)
    draw.text((x + width / 2, y + 12), text, font=fnt, fill=color, anchor="mm")
    return width


def draw_earth(draw, cx, cy, r):
    for rr in range(r, 0, -1):
        p = 1 - rr / r
        shade = int(78 + 112 * p)
        draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=(shade, shade, shade))
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(225, 225, 225), width=2)
    draw.ellipse((cx - r * .96, cy - r * .37, cx + r * .96, cy + r * .37), outline=(155, 155, 155), width=1)
    draw.ellipse((cx - r * .43, cy - r * .96, cx + r * .43, cy + r * .96), outline=(155, 155, 155), width=1)


def make_frame(frame_index: int) -> Image.Image:
    im = Image.new("RGB", (W, H), (5, 5, 5))
    d = ImageDraw.Draw(im)
    d.text((38, 25), "SpaceFL · 太空联邦学习", font=F28, fill=(242, 242, 242))
    d.text((40, 67), "本地训练持续进行　·　仅在通信窗口传输模型更新", font=F14, fill=(153, 153, 153))
    d.text((W - 34, 34), "概念示意 · 非真实轨道比例", font=F12, fill=(120, 120, 120), anchor="ra")

    cx, cy = 600, 355
    earth_r, orbit_rx, orbit_ry = 103, 360, 184
    station_x = cx + math.cos(STATION_ANGLE) * earth_r
    station_y = cy + math.sin(STATION_ANGLE) * earth_r
    d.ellipse((cx - orbit_rx, cy - orbit_ry, cx + orbit_rx, cy + orbit_ry), outline=(77, 77, 77), width=2)
    draw_earth(d, cx, cy, earth_r)

    # Place the ground station directly on the globe's visible surface.
    nx, ny = math.cos(STATION_ANGLE), math.sin(STATION_ANGLE)
    base_x, base_y = station_x + nx * 3, station_y + ny * 3
    tip_x, tip_y = base_x + nx * 22, base_y + ny * 22
    perp_x, perp_y = -ny, nx
    d.line((base_x, base_y, tip_x, tip_y), fill=(240, 240, 240), width=3)
    d.line((base_x + perp_x * 8, base_y + perp_y * 8, base_x - perp_x * 8, base_y - perp_y * 8), fill=(240, 240, 240), width=2)
    dish_x, dish_y = tip_x + nx * 4, tip_y + ny * 4
    d.arc((dish_x - 8, dish_y - 8, dish_x + 8, dish_y + 8), 200, 340, fill=(240, 240, 240), width=2)
    d.ellipse((station_x - 4, station_y - 4, station_x + 4, station_y + 4), fill=(255, 255, 255))
    draw_label(d, int(station_x + nx * 29), int(station_y + ny * 24), "地面站", "#eeeeee")

    orbiters = []
    for i in range(N_SATS):
        phase = i * math.tau / N_SATS + .2
        speed = math.tau / FRAMES * (1 + (i % 3) * .035)
        angle = phase + frame_index * speed
        sx, sy = cx + math.cos(angle) * orbit_rx, cy + math.sin(angle) * orbit_ry
        orbiters.append((i, angle, sx, sy))

    visible = [item for item in orbiters if abs((item[1] - STATION_ANGLE + math.pi) % math.tau - math.pi) < .29]
    for i, angle, sx, sy in orbiters:
        # Local-computation pulse runs continuously on every spacecraft.
        pulse = 13 + 4 * (0.5 + 0.5 * math.sin(frame_index * .45 + i))
        d.arc((sx - pulse, sy - pulse, sx + pulse, sy + pulse), 25, 300, fill=(103, 149, 255), width=2)
        d.rectangle((sx - 7, sy - 5, sx + 7, sy + 5), fill=(224, 224, 224), outline=(20, 20, 20))
        d.rectangle((sx - 15, sy - 3, sx - 8, sy + 3), fill=(112, 134, 154))
        d.rectangle((sx + 8, sy - 3, sx + 15, sy + 3), fill=(112, 134, 154))
        d.text((sx, sy - 19), f"S{i + 1}", font=F12, fill=(178, 178, 178), anchor="mm")

    if visible:
        i, angle, sx, sy = visible[0]
        d.line((station_x, station_y, sx, sy), fill=(85, 227, 155), width=3)
        for k in range(4):
            q = (frame_index * .075 + k * .25) % 1
            px, py = station_x + (sx - station_x) * q, station_y + (sy - station_y) * q
            d.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(85, 227, 155))
        mx, my = (station_x + sx) / 2, (station_y + sy) / 2
        draw_label(d, int(mx + 13), int(my - 32), "通信窗口：模型更新", "#55e39b")

    draw_label(d, 42, 193, "星上本地训练", "#8bb5ff")
    d.text((43, 231), "各卫星独立使用本地数据，训练过程持续进行", font=F12, fill=(160, 160, 160))
    caption = f"S{visible[0][0] + 1}进入可见窗口，正在向地面站发送模型更新" if visible else "卫星继续本地训练，等待进入地面站可见窗口"
    d.text((W / 2, H - 33), caption, font=F14, fill=(205, 205, 205), anchor="mm")
    # Small color key keeps the one-scene explanation self-contained.
    d.ellipse((42, H - 78, 52, H - 68), fill=(110, 157, 255))
    d.text((61, H - 73), "星上训练", font=F12, fill=(165, 165, 165), anchor="lm")
    d.ellipse((143, H - 78, 153, H - 68), fill=(85, 227, 155))
    d.text((162, H - 73), "模型传输", font=F12, fill=(165, 165, 165), anchor="lm")
    return im.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [make_frame(i) for i in range(FRAMES)]
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=DELAY, loop=0, optimize=False)
    print(OUT)


if __name__ == "__main__":
    main()
