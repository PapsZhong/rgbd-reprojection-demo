import csv
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size, bold=False):
    name = 'segoeuib.ttf' if bold else 'segoeui.ttf'
    windows_font = Path('C:/Windows/Fonts') / name
    if windows_font.exists():
        return ImageFont.truetype(str(windows_font), size)
    return ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf', size)


def main(results):
    with (results / 'metrics.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    motion = [row for row in rows if float(row['shift']) != 0 or float(row['yaw_deg']) != 0]
    panels = [
        ('reference', 'Target reference', 'Direct rendering from the target camera'),
        ('nearest_z', 'Nearest + z-buffer', 'One destination pixel per sample'),
        ('bilinear', 'Bilinear splatting', 'Four-pixel weights, all depths blended'),
        ('depth_gated', 'Depth-gated splatting', 'Rejects samples behind the front surface'),
    ]
    canvas = Image.new('RGB', (1464, 719), '#f4f1ea')
    draw = ImageDraw.Draw(canvas)
    ink, muted, accent = '#172a2c', '#546461', '#116863'
    draw.text((30, 21), 'RGB-D novel-view reprojection', font=font(35, True), fill=ink)
    draw.text((31, 67), 'Synthetic scene 1  /  Shift -0.14 scene units  /  Yaw -2 degrees', font=font(17), fill=muted)
    for i, (key, title, caption) in enumerate(panels):
        left = 30 + i * 353
        draw.text((left, 106), title, font=font(21, True), fill=ink)
        draw.text((left, 135), caption, font=font(14), fill=muted)
        with Image.open(results / f'{key}.tga') as original:
            frame = original.convert('RGB')
        canvas.paste(frame.resize((336, 240), Image.Resampling.LANCZOS), (left, 160))
        draw.rectangle((left + 96, 160 + 67, left + 238, 160 + 157), outline='#faf2dd', width=2)
        canvas.paste(frame.crop((128, 90, 318, 210)).resize((336, 212), Image.Resampling.NEAREST), (left, 421))
        if key == 'reference':
            draw.text((left, 650), 'Top row: full target image', font=font(17, True), fill=accent)
            draw.text((left, 677), 'Bottom row: shared crop', font=font(15), fill=muted)
        else:
            group = [row for row in motion if row['method'] == key]
            holes = statistics.mean(float(row['hole_pct']) for row in group)
            psnr = statistics.mean(float(row['common_valid_psnr_db']) for row in group)
            draw.text((left, 650), f'{holes:.2f}% holes  /  {psnr:.2f} dB', font=font(17, True), fill=accent)
            draw.text((left, 677), 'Mean of six motion cases, shared-mask PSNR', font=font(13), fill=muted)
    canvas.save(results / 'preview.png')

def multiview_preview(results):
    with (results / 'multiview.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    count = len({row['target'] for row in rows})
    panels = [
        ('target', 'Held-out target', 'Captured RGB frame'),
        ('near_single', 'Nearest single view', 'Depth-gated reprojection'),
        ('joint_splat', 'Joint depth-gated splat', 'Two source views'),
        ('fill_only', 'Auxiliary hole fill', 'Preserves the nearest view'),
    ]
    canvas = Image.new('RGB', (1422, 431), '#f4f1ea')
    draw = ImageDraw.Draw(canvas)
    ink, muted, accent = '#172a2c', '#546461', '#116863'
    draw.text((24, 15), 'Fountain RGB-D reconstruction', font=font(30, True), fill=ink)
    draw.text((25, 55), f'Example target  /  Mean metrics over {count} held-out frames',
              font=font(16), fill=muted)
    for i, (key, title, caption) in enumerate(panels):
        left = 24 + i * 347
        draw.text((left, 88), title, font=font(20, True), fill=ink)
        draw.text((left, 114), caption, font=font(14), fill=muted)
        with Image.open(results / f'{key}.tga') as original:
            frame = original.convert('RGB').resize((333, 250), Image.Resampling.LANCZOS)
        canvas.paste(frame, (left, 139))
        if key == 'target':
            continue
        group = [row for row in rows if row['method'] == key]
        holes = statistics.mean(float(row['hole_pct']) for row in group)
        psnr = statistics.mean(float(row['shared_psnr_db']) for row in group)
        draw.text((left, 398), f'{holes:.2f}% holes  /  {psnr:.2f} dB shared-mask PSNR',
                  font=font(15, True), fill=accent)
    canvas.save(results / 'preview.png')


if __name__ == '__main__':
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results')
    if (results / 'multiview.csv').is_file():
        multiview_preview(results)
    else:
        main(results)
