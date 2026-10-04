"""
AI-Based Food Image Nutrition Estimation System
Food Detection • Segmentation • Portion Estimation • Nutrition Analysis

Interactive Streamlit Application implementing the full end-to-end inference pipeline:
  Image -> Preprocessing -> Detection -> Segmentation -> Classification ->
  Ingredients -> Portion/Mass/Volume -> Multi-Nutrient Estimation -> Aggregation -> Quad Output
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from inference.predict import format_detailed_cli_report, predict_meal
from src.nutrition.visualizer import (
    draw_detected_food_image,
    draw_meal_predictions,
    draw_segmented_food_image,
)
from src.utils.csv_export import export_predictions_csv
from src.utils.html_report import generate_html_report


st.set_page_config(
    page_title="AI-Based Food Image Nutrition Estimation System",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.0rem;
    }
    .disclaimer-banner {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.92rem;
        color: #92400E;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 12px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-lbl {
        font-size: 0.80rem;
        color: #64748B;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .metric-val {
        font-size: 1.9rem;
        font-weight: 800;
        margin-top: 4px;
    }
    .item-chip {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .badge-prediction {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-derived {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-proxy {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_demo_presets() -> list[dict[str, Any]]:
    preset_file = PROJECT_ROOT / "data/demo_meals/demo_meals.json"
    if preset_file.is_file():
        with open(preset_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def plot_macronutrient_donut(protein_g: float, carbs_g: float, fat_g: float) -> plt.Figure:
    """Generate clean macronutrient energy distribution donut chart."""
    # NaN and infinity can arise from incomplete nutrition predictions and
    # produce invalid wedge angles in Matplotlib.
    def finite_nonnegative(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return max(0.0, number) if np.isfinite(number) else 0.0

    protein_g, carbs_g, fat_g = map(finite_nonnegative, (protein_g, carbs_g, fat_g))
    cal_p = protein_g * 4.0
    cal_c = carbs_g * 4.0
    cal_f = fat_g * 9.0

    labels = ["Protein (4 kcal/g)", "Carbs (4 kcal/g)", "Fat (9 kcal/g)"]
    sizes = [cal_p, cal_c, cal_f]
    colors = ["#3B82F6", "#F59E0B", "#EF4444"]

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    if sum(sizes) > 0:
        _, _, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.75,
            textprops={"fontsize": 9, "color": "#1E293B"},
        )
        for at in autotexts:
            at.set_color("white")
            at.set_weight("bold")
    else:
        ax.text(0, 0, "No macro\ndata available", ha="center", va="center", fontsize=10, color="#64748B")

    centre_circle = plt.Circle((0, 0), 0.55, fc="white")
    fig.gca().add_artist(centre_circle)
    ax.axis("equal")
    plt.tight_layout()
    return fig


def main():
    presets = load_demo_presets()

    # -------------------------------------------------------------------------
    # Header & Mandatory Disclaimer
    # -------------------------------------------------------------------------
    st.markdown('<div class="main-title">🥗 AI-Based Food Image Nutrition Estimation System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Food Detection • Segmentation • Portion Estimation • Nutrition Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="disclaimer-banner">
            ⚠️ <strong>Important:</strong> Nutritional values are model estimates and may differ from actual nutritional values.
            Unsupported nutrients are not presented as exact measurements.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Sidebar: Configurations & Options
    # -------------------------------------------------------------------------
    st.sidebar.markdown("### ⚙️ System Controls")
    input_source = st.sidebar.radio(
        "Select Image Input Mode",
        options=["Upload Food / Meal Image", "Select Pre-configured Demo Meal"],
        index=0,
    )

    image_path: str | None = None
    depth_path: str | None = None
    reference_bbox: list[float] | None = None
    reference_real_size_cm: float = 2.5
    preconfigured_boxes = None

    temp_dir = PROJECT_ROOT / "outputs/temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    if input_source == "Upload Food / Meal Image":
        uploaded_file = st.file_uploader(
            "1. Upload food/meal image (JPEG, PNG, WebP):",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload any meal photo taken from phone, camera, or dataset.",
        )
        if uploaded_file is not None:
            save_path = temp_dir / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_path = str(save_path)

        with st.sidebar.expander("Advanced Sensor Inputs (Depth & Calibration Coin)", expanded=False):
            up_depth = st.file_uploader("Upload Depth Map (Optional)", type=["png", "tif", "jpg"])
            if up_depth:
                d_path = temp_dir / f"depth_{up_depth.name}"
                with open(d_path, "wb") as f:
                    f.write(up_depth.getbuffer())
                depth_path = str(d_path)

            has_ref = st.checkbox("Calibrated Reference Object Present", value=False)
            if has_ref:
                reference_real_size_cm = st.number_input("Ref Object Real Size (cm)", min_value=0.5, max_value=20.0, value=2.5, step=0.1)
                r_box = st.text_input("Reference Box [x1, y1, x2, y2]", value="30, 30, 80, 80")
                try:
                    reference_bbox = [float(v.strip()) for v in r_box.split(",")]
                except Exception:
                    reference_bbox = None

    else:
        if presets:
            preset_dict = {p["title"]: p for p in presets}
            selected_preset_title = st.selectbox(
                "Choose Demo Meal Preset:",
                list(preset_dict.keys()),
            )
            p_data = preset_dict[selected_preset_title]
            image_path = p_data["image_path"]
            depth_path = p_data.get("depth_path")
            reference_bbox = p_data.get("reference_bbox")
            reference_real_size_cm = p_data.get("reference_size_cm", 2.5)
            preconfigured_boxes = p_data.get("preconfigured_boxes")
            st.caption(f"ℹ️ Selected: {selected_preset_title} ({'RGB-D depth available' if depth_path else 'RGB monocular'})")
        else:
            st.warning("No demo presets found.")

    # -------------------------------------------------------------------------
    # Image Preview & Analyze Action Button
    # -------------------------------------------------------------------------
    if not image_path or not Path(image_path).is_file():
        st.info("👆 Please upload a meal photo or select a demo preset above to begin.")
        return

    st.markdown("### 2. Preview Image")
    col_prev, col_btn = st.columns([2, 1])
    with col_prev:
        preview_img = Image.open(image_path)
        st.image(preview_img, caption=f"Selected Meal Image ({preview_img.width} × {preview_img.height} px)", use_container_width=True)

    with col_btn:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("#### Ready to inspect:")
        st.markdown("- Image format: Verified")
        st.markdown(f"- Dimensions: `{preview_img.width} x {preview_img.height}` px")
        st.markdown(f"- 3D Depth Map: `{'Yes' if depth_path else 'None'}`")
        st.markdown(f"- Reference Marker: `{'Yes (2.5cm)' if reference_bbox else 'None'}`")
        analyze_clicked = st.button("🔍 3. Analyze Meal Nutrition", type="primary", use_container_width=True)

    # -------------------------------------------------------------------------
    # Inference Execution
    # -------------------------------------------------------------------------
    if "analysis_results" not in st.session_state or analyze_clicked or st.session_state.get("analyzed_image") != image_path:
        if analyze_clicked or "analysis_results" not in st.session_state:
            with st.spinner("Executing Complete Inference Pipeline (Detection ➔ Segmentation ➔ Classification ➔ Portion ➔ Multi-Nutrients)..."):
                results = predict_meal(
                    image_path=image_path,
                    depth_path=depth_path,
                    reference_bbox=reference_bbox,
                    reference_real_size_cm=reference_real_size_cm,
                    preconfigured_boxes=preconfigured_boxes,
                )
                st.session_state["analysis_results"] = results
                st.session_state["analyzed_image"] = image_path
        else:
            return
    else:
        results = st.session_state["analysis_results"]

    meal_total = results.get("meal_total", {})
    instances = results.get("instances", [])
    num_foods = results.get("num_foods", len(instances))

    st.markdown("---")
    st.markdown("## 4. Analysis Results")

    # -------------------------------------------------------------------------
    # Prominently Displayed TOTALS
    # -------------------------------------------------------------------------
    def _val(item: Any, default: float = 0.0) -> float:
        if item is None:
            return default
        if isinstance(item, dict):
            item = item.get("value")
        try:
            value = float(item) if item is not None else default
            return value if np.isfinite(value) else default
        except (ValueError, TypeError):
            return default

    tot_mass = _val(meal_total.get("total_mass_g", meal_total.get("mass_g", 0.0)))
    tot_cals = _val(meal_total.get("total_calories", meal_total.get("calories", 0.0)))
    tot_prot = _val(meal_total.get("total_protein", meal_total.get("protein_g", 0.0)))
    tot_carb = _val(meal_total.get("total_carbohydrates", meal_total.get("carbs_g", 0.0)))
    tot_fat = _val(meal_total.get("total_fat", meal_total.get("fat_g", 0.0)))

    st.markdown("### 🏆 Total Meal Nutrition Summary")
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">TOTAL MASS</div>
                <div class="metric-val" style="color: #059669;">{tot_mass:.1f} <span style="font-size:1rem;">g</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">TOTAL CALORIES</div>
                <div class="metric-val" style="color: #EA580C;">{tot_cals:.0f} <span style="font-size:1rem;">kcal</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">TOTAL PROTEIN</div>
                <div class="metric-val" style="color: #2563EB;">{tot_prot:.1f} <span style="font-size:1rem;">g</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">TOTAL CARBOHYDRATES</div>
                <div class="metric-val" style="color: #D97706;">{tot_carb:.1f} <span style="font-size:1rem;">g</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">TOTAL FAT</div>
                <div class="metric-val" style="color: #DC2626;">{tot_fat:.1f} <span style="font-size:1rem;">g</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Visualizations: 4 Views
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🖼️ Pipeline Visualizations")

    det_path = results.get("detected_image_path")
    seg_path = results.get("segmented_image_path")
    ann_path = results.get("annotated_image_path")

    v1, v2, v3, v4 = st.columns(4)
    with v1:
        st.markdown("#### 1. Original Image")
        st.image(image_path, use_container_width=True)

    with v2:
        st.markdown("#### 2. Detected Food Image")
        if det_path and Path(det_path).is_file():
            st.image(det_path, use_container_width=True)
        else:
            det_img = draw_detected_food_image(image_path, instances)
            st.image(det_img, use_container_width=True)

    with v3:
        st.markdown("#### 3. Segmented Image")
        if seg_path and Path(seg_path).is_file():
            st.image(seg_path, use_container_width=True)
        else:
            st.image(image_path, use_container_width=True)

    with v4:
        st.markdown("#### 4. Final Annotated Image")
        if ann_path and Path(ann_path).is_file():
            st.image(ann_path, use_container_width=True)
        else:
            st.image(image_path, use_container_width=True)

    # -------------------------------------------------------------------------
    # Per-Item Summary Cards
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔍 Detected Food Items Overview")
    if not instances:
        st.info("No food items detected in this image.")
    else:
        item_cols = st.columns(min(len(instances), 4))
        for idx, inst in enumerate(instances):
            col_target = item_cols[idx % len(item_cols)]
            name = inst.get("food_name", "Unknown").replace("_", " ").title()
            conf = float(inst.get("confidence", 0.0)) * 100
            mass = _val(inst.get("estimated_mass_g", 0.0))
            with col_target:
                st.markdown(
                    f"""
                    <div class="item-chip">
                        <div style="font-size:1.05rem; font-weight:700; color:#1E293B;">🍽️ {name}</div>
                        <div style="font-size:0.9rem; color:#475569; margin-top:4px;">
                            <strong>Confidence:</strong> <span class="badge-prediction">{conf:.1f}%</span><br>
                            <strong>Estimated Mass:</strong> <span style="font-weight:700; color:#059669;">{mass:.1f} g</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------------------------------
    # Primary Nutrition Table
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Nutrition Table")

    table_data = []
    for inst in instances:
        name = inst.get("food_name", "Unknown").replace("_", " ").title()
        mass = _val(inst.get("estimated_mass_g", 0.0))
        cals = _val(inst.get("calories_kcal", 0.0))
        prot = _val(inst.get("protein_g", 0.0))
        carb = _val(inst.get("carbohydrates_g", 0.0))
        fat = _val(inst.get("fat_g", 0.0))

        table_data.append({
            "Food": name,
            "Mass (g)": f"{mass:.1f}",
            "Calories (kcal)": f"{cals:.1f}",
            "Protein (g)": f"{prot:.1f}",
            "Carbohydrates (g)": f"{carb:.1f}",
            "Fat (g)": f"{fat:.1f}",
        })

    # Summary row
    table_data.append({
        "Food": "TOTAL MEAL",
        "Mass (g)": f"{tot_mass:.1f}",
        "Calories (kcal)": f"{tot_cals:.1f}",
        "Protein (g)": f"{tot_prot:.1f}",
        "Carbohydrates (g)": f"{tot_carb:.1f}",
        "Fat (g)": f"{tot_fat:.1f}",
    })

    df_primary = pd.DataFrame(table_data)
    st.dataframe(df_primary, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # Additional Supported Nutrients
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔬 Additional Supported Nutrients (Fiber, Sugar, Micronutrients)", expanded=True):
        st.markdown("Extended nutrients are scientifically derived via mass-scaled USDA FoodData Central reference laboratory profiles:")

        ext_rows = []
        tot_fiber = _val(meal_total.get("total_fiber_g", meal_total.get("fiber_g")))
        tot_sugar = _val(meal_total.get("total_sugar_g", meal_total.get("sugar_g")))
        tot_satfat = _val(meal_total.get("total_saturated_fat_g", meal_total.get("saturated_fat_g")))
        tot_sodium = _val(meal_total.get("total_sodium_mg", meal_total.get("sodium_mg")))
        tot_chol = _val(meal_total.get("total_cholesterol_mg", meal_total.get("cholesterol_mg")))

        add_nuts = meal_total.get("additional_nutrients", {})
        tot_k = _val(add_nuts.get("potassium_mg"))
        tot_ca = _val(add_nuts.get("calcium_mg"))
        tot_fe = _val(add_nuts.get("iron_mg"))
        tot_vitc = _val(add_nuts.get("vitamin_c_mg"))

        extended_specs = [
            ("Dietary Fiber", tot_fiber, "g"),
            ("Total Sugars", tot_sugar, "g"),
            ("Saturated Fat", tot_satfat, "g"),
            ("Sodium", tot_sodium, "mg"),
            ("Cholesterol", tot_chol, "mg"),
            ("Potassium", tot_k, "mg"),
            ("Calcium", tot_ca, "mg"),
            ("Iron", tot_fe, "mg"),
            ("Vitamin C", tot_vitc, "mg"),
        ]

        for nutrient_name, amount, unit in extended_specs:
            ext_rows.append({
                "Nutrient": nutrient_name,
                "Amount": f"{amount:.1f} {unit}" if amount > 0.0 else "Not available",
                "Unit": unit,
                "Provenance Source": "DERIVED (USDA Reference)" if amount > 0.0 else "NOT AVAILABLE",
            })

        df_ext = pd.DataFrame(ext_rows)
        st.dataframe(df_ext, use_container_width=True, hide_index=True)

        st.markdown("#### Macronutrient Energy Distribution")
        col_chart, col_chart_txt = st.columns([1.2, 1.8])
        with col_chart:
            fig = plot_macronutrient_donut(tot_prot, tot_carb, tot_fat)
            st.pyplot(fig)
        with col_chart_txt:
            cal_p = (tot_prot if np.isfinite(tot_prot) else 0.0) * 4.0
            cal_c = (tot_carb if np.isfinite(tot_carb) else 0.0) * 4.0
            cal_f = (tot_fat if np.isfinite(tot_fat) else 0.0) * 9.0
            cal_sum = max(1.0, cal_p + cal_c + cal_f)
            st.markdown(
                f"""
                - **Protein Energy:** `{cal_p:.1f} kcal` ({cal_p / cal_sum * 100:.1f}%)
                - **Carbohydrates Energy:** `{cal_c:.1f} kcal` ({cal_c / cal_sum * 100:.1f}%)
                - **Fat Energy:** `{cal_f:.1f} kcal` ({cal_f / cal_sum * 100:.1f}%)
                
                *Calculated via standard Atwater factors: Protein (4 kcal/g), Carbohydrates (4 kcal/g), Fat (9 kcal/g).*
                """
            )

    # -------------------------------------------------------------------------
    # Report & Export Actions
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💾 Report & Data Export")

    json_str = json.dumps(results, indent=2)
    csv_str = ""
    csv_path = results.get("csv_path")
    if csv_path and Path(csv_path).is_file():
        csv_str = Path(csv_path).read_text(encoding="utf-8")
    else:
        temp_csv = temp_dir / "temp_pred.csv"
        export_predictions_csv(results, temp_csv)
        csv_str = temp_csv.read_text(encoding="utf-8")

    html_report_str = generate_html_report(results)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name="predictions.json",
            mime="application/json",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            label="📊 Download CSV",
            data=csv_str,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            label="📄 Download HTML / PDF Report",
            data=html_report_str,
            file_name="nutrition_report.html",
            mime="text/html",
            use_container_width=True,
            help="Download standalone HTML report. Open in browser and click 'Print / Save as PDF' to save as PDF.",
        )


if __name__ == "__main__":
    main()
