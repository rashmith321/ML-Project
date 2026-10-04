"""
HTML/PDF Report Generator for AI-Based Food Image Nutrition Estimation System.

Generates self-contained, beautifully styled HTML reports with print-to-PDF
support, responsive tables, provenance badges, and mandatory scientific disclaimers.
"""
from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Any


def _to_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, dict):
        v = v.get("value")
    try:
        return float(v) if v is not None else default
    except (ValueError, TypeError):
        return default


def generate_html_report(payload: dict[str, Any]) -> str:
    """
    Renders an HTML report document with clean print/PDF styling.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_path = payload.get("image_path", "Unknown")
    img_name = Path(image_path).name if image_path else "food_image"
    instances = payload.get("instances", [])
    num_foods = payload.get("num_foods", len(instances))
    meal_total = payload.get("meal_total", {})

    # Extract meal totals
    tot_mass = _to_float(meal_total.get("total_mass_g", meal_total.get("mass_g", 0.0)))
    tot_cals = _to_float(meal_total.get("total_calories", meal_total.get("calories", 0.0)))
    tot_prot = _to_float(meal_total.get("total_protein", meal_total.get("protein_g", 0.0)))
    tot_carb = _to_float(meal_total.get("total_carbohydrates", meal_total.get("carbs_g", 0.0)))
    tot_fat = _to_float(meal_total.get("total_fat", meal_total.get("fat_g", 0.0)))

    # Main table rows
    table_rows = []
    for inst in instances:
        name = inst.get("food_name", "Unknown").replace("_", " ").title()
        mass = _to_float(inst.get("estimated_mass_g", 0.0))
        cals = _to_float(inst.get("calories_kcal", 0.0))
        prot = _to_float(inst.get("protein_g", 0.0))
        carb = _to_float(inst.get("carbohydrates_g", 0.0))
        fat = _to_float(inst.get("fat_g", 0.0))

        table_rows.append(f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td>{mass:.1f}</td>
            <td>{cals:.1f}</td>
            <td>{prot:.1f}</td>
            <td>{carb:.1f}</td>
            <td>{fat:.1f}</td>
        </tr>
        """)

    # Extended nutrients
    tot_fiber = _to_float(meal_total.get("total_fiber_g", meal_total.get("fiber_g")))
    tot_sugar = _to_float(meal_total.get("total_sugar_g", meal_total.get("sugar_g")))
    tot_satfat = _to_float(meal_total.get("total_saturated_fat_g", meal_total.get("saturated_fat_g")))
    tot_sodium = _to_float(meal_total.get("total_sodium_mg", meal_total.get("sodium_mg")))
    tot_chol = _to_float(meal_total.get("total_cholesterol_mg", meal_total.get("cholesterol_mg")))

    add_nuts = meal_total.get("additional_nutrients", {})
    tot_k = _to_float(add_nuts.get("potassium_mg"))
    tot_ca = _to_float(add_nuts.get("calcium_mg"))
    tot_fe = _to_float(add_nuts.get("iron_mg"))
    tot_vitc = _to_float(add_nuts.get("vitamin_c_mg"))

    # Per-item cards
    item_cards = []
    for i, inst in enumerate(instances, start=1):
        name = inst.get("food_name", "Unknown").replace("_", " ").title()
        conf = float(inst.get("confidence", 0.0)) * 100
        mass = _to_float(inst.get("estimated_mass_g", 0.0))
        vol = _to_float(inst.get("estimated_volume", 0.0))
        portion = inst.get("portion", {})
        method = portion.get("method_used", "rgb_learned")
        mass_src = portion.get("mass_source", "prediction")
        ingr = inst.get("ingredients", {}).get("candidates", [])
        ingr_src = inst.get("ingredients", {}).get("source", "unavailable")
        ingr_str = ", ".join(ingr) if ingr else "None identified"

        seg = inst.get("segmentation", {})
        seg_area = seg.get("mask_area_px", 0) if isinstance(seg, dict) else 0
        seg_cov = seg.get("mask_coverage_pct", 0.0) if isinstance(seg, dict) else 0.0

        item_cards.append(f"""
        <div class="item-card">
            <h3>🍽️ Item #{i}: {name} <span class="badge badge-conf">{conf:.1f}% Confidence</span></h3>
            <p><strong>Portion Mass:</strong> {mass:.1f} g (Method: <code>{method}</code>, Source: <code>{mass_src}</code>)</p>
            <p><strong>Estimated Volume:</strong> {vol:.1f} cm³ | <strong>Mask Area:</strong> {seg_area:,} px ({seg_cov:.1f}% image area)</p>
            <p><strong>Ingredients Identified:</strong> {ingr_str} <span class="badge badge-src">Source: {ingr_src}</span></p>
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI-Based Food Image Nutrition Estimation System Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
            background-color: #f8fafc;
            margin: 0;
            padding: 30px;
        }}
        .report-container {{
            max-width: 900px;
            margin: 0 auto;
            background: #ffffff;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }}
        .header {{
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 20px;
            margin-bottom: 25px;
        }}
        .title {{
            font-size: 26px;
            font-weight: 800;
            color: #0f172a;
            margin: 0 0 6px 0;
        }}
        .subtitle {{
            font-size: 15px;
            color: #475569;
            margin: 0 0 12px 0;
        }}
        .meta-bar {{
            font-size: 13px;
            color: #64748b;
        }}
        .disclaimer-box {{
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 13px;
            color: #92400e;
            margin: 20px 0;
            font-weight: 500;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
            margin: 25px 0;
        }}
        .metric-box {{
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 14px 10px;
            text-align: center;
        }}
        .metric-label {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            color: #64748b;
            letter-spacing: 0.5px;
        }}
        .metric-val {{
            font-size: 22px;
            font-weight: 800;
            margin-top: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background-color: #f8fafc;
            color: #475569;
            font-weight: 600;
        }}
        tr.total-row td {{
            background-color: #f1f5f9;
            font-weight: 800;
            border-top: 2px solid #cbd5e1;
            border-bottom: 2px solid #cbd5e1;
        }}
        .item-card {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 14px;
        }}
        .item-card h3 {{
            margin: 0 0 10px 0;
            font-size: 16px;
            color: #1e293b;
        }}
        .item-card p {{
            margin: 4px 0;
            font-size: 13px;
            color: #334155;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-conf {{
            background-color: #dbeafe;
            color: #1e40af;
        }}
        .badge-src {{
            background-color: #d1fae5;
            color: #065f46;
        }}
        .footer {{
            margin-top: 35px;
            padding-top: 15px;
            border-top: 1px solid #e2e8f0;
            font-size: 12px;
            color: #94a3b8;
            text-align: center;
        }}
        @media print {{
            body {{
                background: #ffffff;
                padding: 0;
            }}
            .report-container {{
                box-shadow: none;
                padding: 15px;
            }}
            .print-btn {{
                display: none;
            }}
        }}
        .print-btn {{
            background: #2563eb;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            float: right;
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <button class="print-btn" onclick="window.print()">🖨️ Print / Save as PDF</button>
        <div class="header">
            <h1 class="title">AI-Based Food Image Nutrition Estimation System</h1>
            <div class="subtitle">Food Detection • Segmentation • Portion Estimation • Nutrition Analysis</div>
            <div class="meta-bar">
                Generated: <strong>{now_str}</strong> | Image: <strong>{img_name}</strong> | Detected Foods: <strong>{num_foods}</strong>
            </div>
        </div>

        <div class="disclaimer-box">
            ⚠️ <strong>Important Notice:</strong> Nutritional values are model estimates and may differ from actual nutritional values. Unsupported nutrients are not presented as exact measurements.
        </div>

        <h2 style="font-size: 18px; margin-top: 20px;">Prominent Meal Summary</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">TOTAL MASS</div>
                <div class="metric-val" style="color: #059669;">{tot_mass:.1f} <span style="font-size: 12px;">g</span></div>
            </div>
            <div class="metric-box">
                <div class="metric-label">TOTAL CALORIES</div>
                <div class="metric-val" style="color: #ea580c;">{tot_cals:.0f} <span style="font-size: 12px;">kcal</span></div>
            </div>
            <div class="metric-box">
                <div class="metric-label">TOTAL PROTEIN</div>
                <div class="metric-val" style="color: #2563eb;">{tot_prot:.1f} <span style="font-size: 12px;">g</span></div>
            </div>
            <div class="metric-box">
                <div class="metric-label">TOTAL CARBS</div>
                <div class="metric-val" style="color: #d97706;">{tot_carb:.1f} <span style="font-size: 12px;">g</span></div>
            </div>
            <div class="metric-box">
                <div class="metric-label">TOTAL FAT</div>
                <div class="metric-val" style="color: #dc2626;">{tot_fat:.1f} <span style="font-size: 12px;">g</span></div>
            </div>
        </div>

        <h2 style="font-size: 18px; margin-top: 30px;">Itemized Nutrition Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Food</th>
                    <th>Mass (g)</th>
                    <th>Calories (kcal)</th>
                    <th>Protein (g)</th>
                    <th>Carbohydrates (g)</th>
                    <th>Fat (g)</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
                <tr class="total-row">
                    <td>TOTAL MEAL</td>
                    <td>{tot_mass:.1f}</td>
                    <td>{tot_cals:.1f}</td>
                    <td>{tot_prot:.1f}</td>
                    <td>{tot_carb:.1f}</td>
                    <td>{tot_fat:.1f}</td>
                </tr>
            </tbody>
        </table>

        <h2 style="font-size: 18px; margin-top: 30px;">Additional Supported Nutrients</h2>
        <table>
            <thead>
                <tr>
                    <th>Extended Nutrient</th>
                    <th>Meal Total</th>
                    <th>Unit</th>
                    <th>Provenance Source</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Dietary Fiber</td>
                    <td>{tot_fiber:.1f}</td>
                    <td>g</td>
                    <td>DERIVED (USDA Reference)</td>
                </tr>
                <tr>
                    <td>Total Sugars</td>
                    <td>{tot_sugar:.1f}</td>
                    <td>g</td>
                    <td>DERIVED (USDA Reference)</td>
                </tr>
                <tr>
                    <td>Saturated Fat</td>
                    <td>{tot_satfat:.1f}</td>
                    <td>g</td>
                    <td>DERIVED (USDA Reference)</td>
                </tr>
                <tr>
                    <td>Sodium</td>
                    <td>{tot_sodium:.1f}</td>
                    <td>mg</td>
                    <td>DERIVED (USDA Reference)</td>
                </tr>
                <tr>
                    <td>Cholesterol</td>
                    <td>{tot_chol:.1f}</td>
                    <td>mg</td>
                    <td>DERIVED (USDA Reference)</td>
                </tr>
                <tr>
                    <td>Potassium</td>
                    <td>{tot_k:.1f}</td>
                    <td>mg</td>
                    <td>DERIVED (USDA Reference)</td>
                </tr>
                <tr>
                    <td>Calcium</td>
                    <td>{tot_ca:.1f}</td>
                    <td>mg</td>
                    <td>DERIVED (USDA Reference)</td>
                </tr>
                <tr>
                    <td>Iron</td>
                    <td>{tot_fe:.1f}</td>
                    <td>mg</td>
                    <td>DERIVED (USDA Reference)</td>
                </tr>
                <tr>
                    <td>Vitamin C</td>
                    <td>{tot_vitc:.1f}</td>
                    <td>mg</td>
                    <td>DERIVED (USDA Reference)</td>
                </tr>
            </tbody>
        </table>

        <h2 style="font-size: 18px; margin-top: 30px;">Per-Food Detail Breakdown</h2>
        {''.join(item_cards)}

        <div class="footer">
            AI-Based Food Image Nutrition Estimation System • Powered by Deep Learning & Computer Vision
        </div>
    </div>
</body>
</html>
"""
    return html_content
