"""
Combine all 5 Kenya Construction Cost Forecast visuals into a single master graphic.
"""

import os
from PIL import Image, ImageDraw, ImageFont

FIG_DIR = r"c:\PROJECTS\housing_project\kenya-construction-cost-forecast\output\figures"
OUT_PATH = os.path.join(FIG_DIR, "master_forecast_dashboard.png")

# Load all 5 images
img1 = Image.open(os.path.join(FIG_DIR, "01_eda_overview.png"))
img2 = Image.open(os.path.join(FIG_DIR, "02_stationarity_analysis.png"))
img3 = Image.open(os.path.join(FIG_DIR, "03_holdout_evaluation.png"))
img4 = Image.open(os.path.join(FIG_DIR, "04_forward_forecast.png"))
img5 = Image.open(os.path.join(FIG_DIR, "05_cost_sensitivity.png"))

CANVAS_WIDTH = 3400
HEADER_HEIGHT = 160
PADDING = 30
BG_COLOR = (248, 249, 250)
BORDER_COLOR = (220, 224, 230)

# Helper to resize keeping aspect ratio
def fit_width(img, target_w):
    w, h = img.size
    target_h = int(h * (target_w / w))
    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

def fit_height(img, target_h):
    w, h = img.size
    target_w = int(w * (target_h / h))
    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

# Row 1: EDA (left) + Stationarity (right)
# Target row height ~ 1100
row1_h = 1150
r1_left = fit_height(img1, row1_h)
r1_right = fit_height(img2, row1_h)

# Scale both to fit width with padding
total_r1_w = r1_left.width + r1_right.width + PADDING * 3
if total_r1_w > CANVAS_WIDTH:
    scale = (CANVAS_WIDTH - PADDING * 3) / (r1_left.width + r1_right.width)
    r1_left = r1_left.resize((int(r1_left.width * scale), int(r1_left.height * scale)), Image.Resampling.LANCZOS)
    r1_right = r1_right.resize((int(r1_right.width * scale), int(r1_right.height * scale)), Image.Resampling.LANCZOS)

# Row 2: Holdout Evaluation (full width)
r2_w = CANVAS_WIDTH - (PADDING * 2)
r2_img = fit_width(img3, r2_w)

# Row 3: Forward Forecast (left) + Cost Sensitivity (right)
row3_h = 950
r3_left = fit_height(img4, row3_h)
r3_right = fit_height(img5, row3_h)
total_r3_w = r3_left.width + r3_right.width + PADDING * 3
if total_r3_w > CANVAS_WIDTH:
    scale = (CANVAS_WIDTH - PADDING * 3) / (r3_left.width + r3_right.width)
    r3_left = r3_left.resize((int(r3_left.width * scale), int(r3_left.height * scale)), Image.Resampling.LANCZOS)
    r3_right = r3_right.resize((int(r3_right.width * scale), int(r3_right.height * scale)), Image.Resampling.LANCZOS)

# Compute total canvas height
canvas_height = (
    HEADER_HEIGHT +
    max(r1_left.height, r1_right.height) + PADDING +
    r2_img.height + PADDING +
    max(r3_left.height, r3_right.height) + PADDING * 2
)

canvas = Image.new("RGB", (CANVAS_WIDTH, canvas_height), BG_COLOR)
draw = ImageDraw.Draw(canvas)

# Header banner
draw.rectangle([(0, 0), (CANVAS_WIDTH, HEADER_HEIGHT)], fill=(26, 82, 118)) # #1a5276

try:
    font_title = ImageFont.truetype("arialbd.ttf", 54)
    font_sub = ImageFont.truetype("arial.ttf", 26)
except:
    font_title = ImageFont.load_default()
    font_sub = ImageFont.load_default()

draw.text((PADDING + 20, 32), "Kenya Construction Cost Forecast & AHP Sensitivity Analysis", fill=(255, 255, 255), font=font_title)
draw.text((PADDING + 20, 102), "Quarterly KNBS Construction Input Price Index (Base Dec 2019 = 100) | ARIMA(0,2,1) vs ETS Model Suite | Q3 2026 - Q2 2027 Projections", fill=(212, 230, 241), font=font_sub)

# Paste Row 1
curr_y = HEADER_HEIGHT + PADDING
canvas.paste(r1_left, (PADDING, curr_y))
canvas.paste(r1_right, (PADDING * 2 + r1_left.width, curr_y))

# Paste Row 2
curr_y += max(r1_left.height, r1_right.height) + PADDING
canvas.paste(r2_img, (PADDING, curr_y))

# Paste Row 3
curr_y += r2_img.height + PADDING
canvas.paste(r3_left, (PADDING, curr_y))
canvas.paste(r3_right, (PADDING * 2 + r3_left.width, curr_y))

# Save output
canvas.save(OUT_PATH, quality=95)
print(f"Master composite saved: {OUT_PATH} (Dimensions: {canvas.size})")
