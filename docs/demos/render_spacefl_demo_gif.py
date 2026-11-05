"""Render the SpaceFL concept animation as a presentation-friendly GIF.

This is an illustrative diagram, not a physical orbit propagator. The HTML
version in the same directory is the preferred live demo; this renderer makes
an easily embeddable fallback for slides.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 960, 540
FRAME_COUNT, FRAME_MS = 72, 90
OUT = Path(__file__).resolve().parents[2] / "output" / "spacefl_federated_learning_demo.gif"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = Path("C:/Windows/Fonts/msyh.ttc")
    if not path.exists():
        path = Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf")
    return ImageFont.truetype(str(path), size)


F12 = font(12)
F14 = font(14)
F16 = font(16, True)
F22 = font(22, True)
F28 = font(28, True)


def blend(a: tuple[int, int, int], b: tuple[int, int, int], p: float) -> tuple[int, int, int]:
    return tuple(int(a[i] * (1 - p) + b[i] * p) for i in range(3))


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt, fill, anchor=None):
    draw.text(xy, text, font=fnt, fill=fill, anchor=anchor)


def frame(frame_no: int) -> Image.Image:
    im = Image.new("RGB", (WIDTH, HEIGHT), (4, 10, 18))
    px = im.load()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            radial = min(1.0, math.hypot(x - WIDTH * .49, y - HEIGHT * .43) / 650)
            px[x, y] = blend((16, 45, 70), (4, 10, 18), radial)
    d = ImageDraw.Draw(im, "RGBA")
    for i in range(85):
        x = (i * 83) % WIDTH
        y = (i * 47) % HEIGHT
        d.ellipse((x, y, x + 2, y + 2), fill=(172, 216, 238, 120))

    cx, cy, earth_r = 480, 238, 70
    for radius, alpha in ((122, 130), (149, 90), (176, 58)):
        d.ellipse((cx - radius, cy - radius * .38, cx + radius, cy + radius * .38), outline=(82, 164, 207, alpha), width=1)
    glow = Image.new("RGBA", im.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((cx - earth_r - 14, cy - earth_r - 14, cx + earth_r + 14, cy + earth_r + 14), fill=(47, 189, 250, 50))
    im = Image.alpha_composite(im.convert("RGBA"), glow)
    d = ImageDraw.Draw(im, "RGBA")
    for rr in range(earth_r, 0, -1):
        p = 1 - rr / earth_r
        d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=(*blend((6, 48, 82), (70, 200, 238), p), 255))
    d.ellipse((cx - earth_r, cy - earth_r, cx + earth_r, cy + earth_r), outline=(150, 239, 255, 210), width=1)
    for i in range(7):
        a = i * 1.7 + .5
        x, y = cx + math.cos(a) * 35, cy + math.sin(a) * 31
        d.ellipse((x - 12, y - 4, x + 12, y + 4), fill=(103, 222, 178, 72))

    sx, sy = 790, 246
    active = False
    states = []
    for i in range(7):
        phase = ((frame_no / FRAME_COUNT) * 4 + i * .61) % 4
        state = 0 if phase < 1.45 else 1 if phase < 2.35 else 2 if phase < 2.9 else 3
        states.append(state)
        active |= state == 2
    d.line((sx - 12, sy + 12, sx, sy - 9, sx + 12, sy + 12), fill=(255, 180, 84, 240) if active else (100, 132, 156, 220), width=2)
    d.arc((sx - 10, sy - 17, sx + 10, sy + 3), 205, 335, fill=(255, 180, 84, 240) if active else (100, 132, 156, 220), width=2)
    d.line((sx, sy - 7, sx, sy + 10), fill=(180, 218, 238, 220), width=1)
    draw_text(d, (sx, sy + 27), "地面站", F12, (159, 195, 218, 255), anchor="mm")

    for i, state in enumerate(states):
        angle = i * math.tau / 7 + .35 + frame_no * (.035 + (i % 4) * .004)
        radius = (122, 149, 176)[i % 3]
        x, y = cx + math.cos(angle) * radius, cy + math.sin(angle) * radius * .38
        if state == 2:
            d.line((x, y, sx, sy), fill=(255, 180, 84, 205), width=2)
        if state in (0, 1):
            progress = ((frame_no * .025 + i * .19) % 1)
            box = 13 + int(progress * 6)
            d.arc((x - box, y - box, x + box, y + box), 0, int(360 * progress), fill=(110, 157, 255, 210), width=2)
        fill = (255, 180, 84, 255) if state == 2 else (96, 224, 165, 255) if state == 3 else (66, 217, 255, 255)
        d.rectangle((x - 7, y - 5, x + 7, y + 5), fill=(220, 239, 251, 255), outline=(36, 72, 98, 255))
        d.rectangle((x - 14, y - 3, x - 8, y + 3), fill=fill)
        d.rectangle((x + 8, y - 3, x + 14, y + 3), fill=fill)
        draw_text(d, (x, y - 17), f"S{i + 1}", F12, (184, 214, 232, 255), anchor="mm")

    d.rounded_rectangle((22, 20, 465, 92), radius=10, fill=(7, 21, 35, 220), outline=(38, 75, 102, 255), width=1)
    draw_text(d, (40, 34), "SPACE FEDERATED LEARNING", F12, (66, 217, 255, 255))
    draw_text(d, (40, 54), "轨道变化下的协同训练", F28, (238, 246, 255, 255))
    draw_text(d, (40, 84), "卫星本地训练 · 通信窗口传输 · 地面站聚合", F12, (168, 190, 211, 255))
    d.rounded_rectangle((692, 28, 934, 58), radius=15, fill=(15, 38, 58, 225), outline=(44, 91, 123, 255), width=1)
    draw_text(d, (813, 43), "概念示意 · 非真实轨道比例", F12, (198, 232, 255, 255), anchor="mm")

    d.rounded_rectangle((22, 397, 272, 512), radius=10, fill=(10, 23, 38, 235), outline=(38, 68, 93, 255), width=1)
    draw_text(d, (39, 414), "节点状态", F16, (217, 237, 255, 255))
    counts = [states.count(0), states.count(2), states.count(3)]
    labels = [("本地训练", (110, 157, 255, 255)), ("通信传输", (255, 180, 84, 255)), ("已完成聚合", (95, 224, 165, 255))]
    for j, ((label, color), count) in enumerate(zip(labels, counts)):
        yy = 441 + j * 18
        d.ellipse((40, yy - 4, 48, yy + 4), fill=color)
        draw_text(d, (60, yy), label, F12, (209, 226, 240, 255), anchor="lm")
        draw_text(d, (245, yy), str(count), F14, (238, 246, 255, 255), anchor="rm")
    draw_text(d, (39, 496), "仅在可见窗口发送模型更新", F12, (168, 190, 211, 255))

    d.rounded_rectangle((688, 397, 934, 512), radius=10, fill=(10, 23, 38, 235), outline=(38, 68, 93, 255), width=1)
    draw_text(d, (705, 414), "FEDLEO LOOP", F12, (66, 217, 255, 255))
    flow = ["卫星本地训练", "等待通信窗口", "上传模型更新", "地面站聚合并广播"]
    active_step = int((frame_no / FRAME_COUNT) * 4) % 4
    for j, label in enumerate(flow):
        yy = 440 + j * 18
        col = (66, 217, 255, 255) if j == active_step else (168, 190, 211, 255)
        d.ellipse((706, yy - 7, 720, yy + 7), fill=col if j == active_step else (16, 43, 66, 255), outline=(55, 105, 139, 255), width=1)
        draw_text(d, (713, yy), str(j + 1), F12, (7, 17, 29, 255) if j == active_step else (66, 217, 255, 255), anchor="mm")
        draw_text(d, (730, yy), label, F12, col, anchor="lm")
    caption = ["卫星节点正在使用本地数据训练模型", "卫星等待下一次星地通信窗口", "满足可见性条件，模型更新正在上传", "地面站完成聚合并向卫星广播新模型"][active_step]
    draw_text(d, (480, 374), caption, F14, (203, 231, 251, 255), anchor="mm")
    return im.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [frame(i) for i in range(FRAME_COUNT)]
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=FRAME_MS, loop=0, optimize=False)
    print(OUT)


if __name__ == "__main__":
    main()
