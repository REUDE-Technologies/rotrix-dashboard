#type: ignore

import io
import os
import html
import base64
from datetime import datetime
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm, mm
from reportlab.lib import colors as rl_colors
from reportlab.lib.colors import Color, HexColor, white, black, gray
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    PageBreak, KeepTogether, Frame, PageTemplate, BaseDocTemplate,
    NextPageTemplate,
)
from reportlab.platypus.tables import Table as RLTable, TableStyle
from reportlab.platypus.flowables import HRFlowable
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.pdfgen import canvas as rl_canvas

from config import (
    SORTED_TABLE_PDF_COLUMN_SHORT_NAMES, REUDE_LOGO_PATH,
    _drop_sorted_table_report_columns,
)
from plotting import _fig_to_image_bytes, _fig_to_base64
from data_loader import clean_file_info_text, parse_file_info_to_table, seconds_to_mmss
try:
    from resource_manager import detect_quality, force_gc
except ImportError:
    detect_quality = None
    force_gc = None

import threading

_REPORT_SEMAPHORE = threading.Semaphore(1)


def build_report_pdf(include_info: bool, selected_graph_keys, selected_table_keys,
                     include_cover_page=True, include_table_of_contents=True, color_scheme="Professional Blue",
                     cover_company_name: str = "", cover_user_name: str = "",
                     cover_logo_path: str | None = None):
    """
    Build a PDF report (A4 portrait, with landscape for sorted table) using pure-Python libraries.

    This function mirrors the structure of the HTML report:
    - Optional cover page and table of contents (cover can show company name in the box)
    - File info / pre-header text
    - Selected graphs (each with its optional table)
    - Selected standalone tables (Sorted Performance Table uses landscape orientation)
    - cover_company_name: if non-empty, drawn on the cover in the box (Verdana 30pt).
    """
    acquired = _REPORT_SEMAPHORE.acquire(timeout=120)
    if not acquired:
        raise RuntimeError("Report generation is busy. Please try again in a moment.")

    try:
        return _build_report_pdf_impl(
            include_info,
            selected_graph_keys,
            selected_table_keys,
            include_cover_page=include_cover_page,
            include_table_of_contents=include_table_of_contents,
            color_scheme=color_scheme,
            cover_company_name=cover_company_name,
            cover_user_name=cover_user_name,
            cover_logo_path=cover_logo_path,
        )
    finally:
        _REPORT_SEMAPHORE.release()


def _build_report_pdf_impl(include_info: bool, selected_graph_keys, selected_table_keys,
                           include_cover_page=True, include_table_of_contents=True, color_scheme="Professional Blue",
                           cover_company_name: str = "", cover_user_name: str = "",
                           cover_logo_path: str | None = None):
    """
    Internal implementation for building a PDF report.
    """
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            BaseDocTemplate,
            PageTemplate,
            Frame,
            NextPageTemplate,
            Paragraph,
            Spacer,
            Table as RLTable,
            TableStyle,
            PageBreak,
            Image as RLImage,
            KeepTogether,
            Flowable,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        class ConditionalSpacer(Spacer):
            """Spacer that shrinks to available height to avoid LayoutError when near page bottom."""
            def wrap(self, availWidth, availHeight):
                h = min(self.height, max(0, availHeight - 1e-6))
                return (availWidth, h)
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT, TA_JUSTIFY
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.colors import HexColor
        from reportlab.lib.utils import ImageReader
    except ImportError:
        st.error(
            "PDF generation requires the 'reportlab' package. "
            "Please install it (e.g. 'pip install reportlab' or 'pip install -r requirements.txt') "
            "and then restart the app."
        )
        return None

    buffer = BytesIO()

    # Detect system resources for adaptive quality
    _quality = detect_quality() if detect_quality else None
    _low_ram = _quality is not None and _quality.tier == "low"
    _skip_toc_pass1 = _quality.skip_toc_pass1 if _quality else False

    # First, determine which PNG cover image to use (same as HTML report).
    # We now always use the neutral ``Cover Page.png`` layout for the cover.
    # If it is missing on disk, we simply skip the full‑bleed cover and let
    # the report render without it so the legacy cover is never used.
    cover_path = None
    use_fullpage_cover = False
    if include_cover_page:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cover_candidate = os.path.join(base_dir, "Cover Page.png")
            if os.path.exists(cover_candidate):
                cover_path = cover_candidate
                use_fullpage_cover = True
        except Exception:
            cover_path = None
            use_fullpage_cover = False

    # Build page templates: an optional full‑bleed cover template (no margins)
    # plus the standard portrait template (with margins) for all inner pages.
    page_templates = []

    if include_cover_page and use_fullpage_cover and cover_path:
        # Cover template: full page frame with an onPage callback that draws
        # the PNG edge‑to‑edge (no margins), then company name in the box if provided.
        cover_frame = Frame(
            0 * mm, 0 * mm, A4[0], A4[1],
            leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0
        )
        _cover_company = (cover_company_name or "").strip()
        _cover_user = (cover_user_name or "").strip()
        _cover_logo_path = (cover_logo_path or "").strip()

        def _cover_on_page(canvas, doc_obj):
            try:
                img = ImageReader(cover_path)
                canvas.drawImage(img, 0, 0, width=A4[0], height=A4[1])
            except Exception:
                pass
            # Optional organization logo on the cover page.
            if _cover_logo_path:
                try:
                    logo_img = ImageReader(_cover_logo_path)
                    # Draw the logo a bit taller and near the top, centered horizontally.
                    logo_w = 90 * mm
                    logo_h = 46 * mm
                    x_center = A4[0] / 2.0
                    x = x_center - logo_w / 2.0
                    # 8 mm margin from the top edge.
                    y = A4[1] - logo_h - 35 * mm
                    canvas.drawImage(
                        logo_img,
                        x,
                        y,
                        width=logo_w,
                        height=logo_h,
                        preserveAspectRatio=True,
                        anchor='c',
                        mask="auto",
                    )
                except Exception:
                    pass
            if _cover_company:
                try:
                    from reportlab.pdfbase import pdfmetrics
                    from reportlab.pdfbase.ttfonts import TTFont
                    bold_registered = False
                    _base = os.path.dirname(os.path.abspath(__file__))
                    for font_dir in ("fonts", "report_assets/fonts", ""):
                        folder = os.path.join(_base, font_dir) if font_dir else _base
                        for name in ("verdanab.ttf", "Verdana Bold.ttf"):
                            path = os.path.join(folder, name)
                            if os.path.isfile(path):
                                pdfmetrics.registerFont(TTFont("Verdana-Bold", path))
                                bold_registered = True
                                break
                        if bold_registered:
                            break
                    if not bold_registered and os.name == "nt":
                        win_fonts = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
                        for name in ("verdanab.ttf", "Verdana Bold.ttf"):
                            path = os.path.join(win_fonts, name)
                            if os.path.isfile(path):
                                pdfmetrics.registerFont(TTFont("Verdana-Bold", path))
                                bold_registered = True
                                break
                    canvas.saveState()
                    if bold_registered:
                        canvas.setFont("Verdana-Bold", 27)
                    else:
                        canvas.setFont("Helvetica-Bold", 27)
                    center_x = A4[0] / 2
                    y_box = 130 * mm  # vertical position of the cover box (from bottom); lowered to align with box
                    # Same dark blue as title and "for" line above
                    canvas.setFillColor(HexColor("#253B80"))
                    canvas.drawCentredString(center_x, y_box, _cover_company)
                    canvas.restoreState()
                except Exception:
                    pass

            # --- Draw "Prepared by" user name and "Prepared on" IST date ---
            try:
                canvas.saveState()
                canvas.setFillColor(HexColor("#253B80"))
                # Position: right-aligned with the label text on the cover PNG
                # "Prepared by :" label ends at ~ 52mm from left, value starts at ~53mm
                x_value = 66.5 * mm
                y_prepared_by = 85 * mm   # approx vertical position of "Prepared by :" on cover
                y_prepared_on = 73 * mm   # approx vertical position of "Prepared on :" on cover

                canvas.setFont("Helvetica-Bold", 20)
                if _cover_user:
                    canvas.drawString(x_value, y_prepared_by, _cover_user)

                # IST timestamp
                from datetime import datetime
                date_str = datetime.now().strftime("%d %B %Y")
                canvas.drawString(x_value, y_prepared_on, date_str)
                canvas.restoreState()
            except Exception:
                pass

        cover_template = PageTemplate(
            id="cover",
            frames=[cover_frame],
            pagesize=A4,
            onPage=_cover_on_page,
        )
        page_templates.append(cover_template)

    # Standard portrait template: A4 with normal margins 2.54 cm (25.4 mm).
    margin_mm = 25.4  # 2.54 cm
    portrait_frame = Frame(
        margin_mm * mm, margin_mm * mm, A4[0] - 2 * margin_mm * mm, A4[1] - 2 * margin_mm * mm,
        leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0
    )

    def _add_page_number_footer(canvas, doc_obj):
        """Draw page number at bottom center of each portrait page."""
        canvas.saveState()
        canvas.setFont("Helvetica", 10)
        canvas.drawCentredString(A4[0] / 2, 15 * mm, str(canvas.getPageNumber()))
        canvas.restoreState()

    portrait_template = PageTemplate(
        id='portrait', frames=[portrait_frame], pagesize=A4, onPage=_add_page_number_footer
    )
    page_templates.append(portrait_template)

    doc = BaseDocTemplate(
        buffer,
        pageTemplates=page_templates,
    )
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    normal_style = styles["BodyText"]

    # Shared dict for two-pass TOC: marker_id -> page number (filled during first build)
    toc_pages_dict = {}

    class TOCMarker(Flowable):
        """Zero-height flowable that records the current page when drawn (for TOC page numbers)."""
        def __init__(self, marker_id):
            Flowable.__init__(self)
            self.marker_id = marker_id
        def draw(self):
            p = self.canv.getPageNumber()
            toc_pages_dict[self.marker_id] = p
        def wrap(self, aW, aH):
            return (0, 0)

    class MarkerPlaceholder(Flowable):
        """Placeholder in content list; replaced by TOCMarker in pass 1, removed in pass 2."""
        def __init__(self, marker_id):
            Flowable.__init__(self)
            self.marker_id = marker_id
        def draw(self):
            pass
        def wrap(self, aW, aH):
            return (0, 0)

    class TocPageNumber(Flowable):
        """Draws only the page number, right-aligned (no leader line)."""
        def __init__(self, page_num, font_name="Helvetica", font_size=10):
            Flowable.__init__(self)
            self.page_num = page_num
            self.font_name = font_name
            self.font_size = font_size
        def draw(self):
            w = self._width
            h = self._height
            y_center = h / 2.0
            self.canv.setFont(self.font_name, self.font_size)
            self.canv.drawRightString(w, y_center - self.font_size * 0.35, str(self.page_num))
        def wrap(self, aW, aH):
            self._width = aW
            self._height = 16
            return (aW, self._height)

    def _make_graph_insight_paragraphs(heading: str, fig_obj):
        """
        Build up to three short bullet-point insights for a graph.
        Where possible these are computed from the actual trace data so that
        the comments are specific to the current motor test.
        Uses the figure's X-axis title so the body text always matches the diagram.
        """
        if fig_obj is None:
            return []

        heading_lower = (heading or "").lower()
        traces = list(getattr(fig_obj, "data", []) or [])

        # Read X-axis title from the figure so insight text matches the diagram.
        try:
            layout = getattr(fig_obj, "layout", None)
            xaxis = getattr(layout, "xaxis", None)
            x_title_obj = getattr(xaxis, "title", None)
            x_title = (getattr(x_title_obj, "text", None) or "") if x_title_obj else ""
            if isinstance(x_title, dict):
                x_title = x_title.get("text", "") or ""
            x_title = str(x_title).strip()
        except Exception:
            x_title = ""
        x_title_lower = x_title.lower()

        def _fmt_x_val(x_at_max):
            """Format x_at_max for bullet text using the graph's actual X-axis label."""
            if "throttle" in x_title_lower or "%" in x_title:
                return f"throttle ≈ {x_at_max:.0f} %"
            if "time" in x_title_lower or "(s)" in x_title_lower or "sec" in x_title_lower:
                return f"time ≈ {x_at_max:.1f} s"
            if "torque" in x_title_lower or "n·m" in x_title_lower or "n*m" in x_title_lower:
                return f"torque ≈ {x_at_max:.2f} N·m"
            if "thrust" in x_title_lower and ("gf" in x_title_lower or "g" in x_title_lower):
                return f"thrust ≈ {x_at_max:.0f} gf"
            # Generic: use axis title and sensible precision
            if not x_title:
                return f"x ≈ {x_at_max:.2f}"
            return f"{x_title} ≈ {x_at_max:.2f}"

        def _numeric_list(values):
            """
            Safely convert a Plotly sequence (list, tuple, numpy array, etc.)
            into a list of floats. Avoids truth-value checks on numpy arrays.
            """
            out = []
            if values is None:
                return out
            try:
                iterable = list(values)
            except Exception:
                iterable = []
            for v in iterable:
                try:
                    out.append(float(v))
                except Exception:
                    continue
            return out

        # Basic per-trace stats: max, location of max, mean.
        # Bar charts use categorical X (e.g. "50-60"); keep raw x for x_at_max, numeric x only for trapz.
        trace_stats = {}
        for tr in traces:
            name = str(getattr(tr, "name", "") or "").strip()
            ys = _numeric_list(getattr(tr, "y", []))
            if not ys:
                continue
            try:
                xs_raw = list(getattr(tr, "x", []) or [])
            except Exception:
                xs_raw = []
            if len(xs_raw) != len(ys):
                continue
            xs = _numeric_list(getattr(tr, "x", []))
            m = max(ys)
            idx = ys.index(m)
            x_at_max = xs_raw[idx]
            mean_y = sum(ys) / len(ys)
            area_trapz = 0.0
            if len(xs) == len(ys) and len(xs) >= 2:
                try:
                    area_trapz = float(np.trapz(ys, xs))
                except Exception:
                    area_trapz = 0.0
            trace_stats[name] = {
                "max_y": m,
                "x_at_max": x_at_max,
                "mean_y": mean_y,
                "area_trapz": area_trapz,
            }

        def _find_trace(candidates):
            for cand in candidates:
                for name in trace_stats.keys():
                    if cand.lower() in name.lower():
                        return name, trace_stats[name]
            return None, None

        bullets: list[str] = []

        # --- Summary graphs (2.1.x) ---
        if "2.1.2. thrust evolution over time across throttle bands" in heading_lower:
            # Line summary: pick the band with highest average thrust.
            if trace_stats:
                best_name = max(trace_stats.items(), key=lambda kv: kv[1]["mean_y"])[0]
                best = trace_stats[best_name]
                bullets.append(f"• Among the throttle bands, '{best_name}' delivers the highest average thrust over the run.")
                bullets.append(f"• Peak thrust in '{best_name}' occurs around {_fmt_x_val(best['x_at_max'])}.")
                bullets.append("• This view helps confirm which throttle bands sustain useful thrust without excessive fluctuation.")
        elif "2.1.1. time spent on each throttle operating range" in heading_lower:
            # Bar chart: total elapsed seconds per band (sum of Δt from sample timestamps).
            if trace_stats:
                _, stats = max(trace_stats.items(), key=lambda kv: kv[1]["max_y"])
                bullets.append(
                    "• Each bar is the total elapsed time spent in that 10% throttle band, computed by summing "
                    "time steps between consecutive samples while the band is active."
                )
                bullets.append(
                    f"• The longest cumulative dwell is in the '{stats['x_at_max']}' band "
                    f"(about {stats['max_y']:.1f} s)."
                )
        elif "2.1.3. vibration area summary by throttle range" in heading_lower:
            if trace_stats:
                worst_name = max(trace_stats.items(), key=lambda kv: kv[1]["mean_y"])[0]
                worst = trace_stats[worst_name]
                bullets.append(f"• Vibration exposure is highest in the '{worst_name}' throttle band on average.")
                bullets.append(f"• Peak vibration within this band reaches its maximum at approximately {_fmt_x_val(worst['x_at_max'])}.")
                bullets.append("• These regions may warrant additional structural checks or isolation for long-duration operation.")
        elif "2.1.3. thrust area summary by throttle range" in heading_lower:
            if trace_stats:
                peak_name, peak_stats = max(trace_stats.items(), key=lambda kv: kv[1]["max_y"])
                int_name, _ = max(
                    trace_stats.items(),
                    key=lambda kv: kv[1].get("area_trapz", 0.0),
                )
                bullets.append(
                    f"• Peak thrust (~{peak_stats['max_y']:.0f} gf) occurs in the '{peak_name}' band "
                    f"(near {_fmt_x_val(peak_stats['x_at_max'])})."
                )
                if int_name != peak_name:
                    bullets.append(
                        f"• The largest time-integrated thrust (colored area) is from '{int_name}'—bands active on both ramp-up and "
                        "ramp-down can exceed the peak band by area even when maximum thrust happens elsewhere."
                    )
                else:
                    bullets.append(
                        "• This band shows both the highest peak thrust and the largest integrated contribution among the ranges plotted."
                    )

        # --- Performance curves (2.2.x) ---
        elif "speed-based performance trends" in heading_lower:
            thrust_name, thrust_stats = _find_trace(["thrust"])
            eff_name, eff_stats = _find_trace(["syseffect", "efficiency"])
            if thrust_stats:
                bullets.append(f"• Maximum recorded thrust is about {thrust_stats['max_y']:.0f} gf near {_fmt_x_val(thrust_stats['x_at_max'])}.")  # type: ignore[arg-type]
            if eff_stats:
                bullets.append(f"• Peak overall efficiency ({eff_name}) occurs around {_fmt_x_val(eff_stats['x_at_max'])}, defining the most energy-efficient band.")  # type: ignore[arg-type]
            if thrust_stats and eff_stats:
                if eff_stats["x_at_max"] < thrust_stats["x_at_max"]:
                    bullets.append("• Above the efficiency peak, extra throttle produces more thrust but at steadily reducing efficiency.")
                else:
                    bullets.append("• In this test, efficiency continues to improve up to the region of maximum thrust.")
        elif "load-based performance trends" in heading_lower:
            thrust_name, thrust_stats = _find_trace(["thrust"])
            eff_name, eff_stats = _find_trace(["syseffect", "efficiency"])
            if thrust_stats:
                bullets.append(f"• Thrust rises with torque and peaks at about {thrust_stats['max_y']:.0f} gf near {_fmt_x_val(thrust_stats['x_at_max'])}.")  # type: ignore[arg-type]
            if eff_stats:
                bullets.append(f"• Highest efficiency is seen around {_fmt_x_val(eff_stats['x_at_max'])}, where the motor converts input power to thrust most effectively.")  # type: ignore[arg-type]
            bullets.append("• This curve helps confirm torque limits that give strong thrust without overloading the drivetrain.")
        elif "power and efficiency trends" in heading_lower:
            power_name, power_stats = _find_trace(["power"])
            eff_name, eff_stats = _find_trace(["syseffect", "efficiency"])
            if power_stats:
                bullets.append(f"• Electrical power draw increases up to roughly {power_stats['max_y']:.0f} W at the highest tested thrust point.")  # type: ignore[arg-type]
            if eff_stats:
                bullets.append(f"• System efficiency peaks near {_fmt_x_val(eff_stats['x_at_max'])}, providing a good target for continuous operation.")  # type: ignore[arg-type]
            bullets.append("• Beyond this region, additional thrust comes at a steeper power cost, which is important for sizing batteries and ESCs.")
        elif "thrust and vibration response" in heading_lower:
            thrust_name, thrust_stats = _find_trace(["thrust"])
            vib_name, vib_stats = _find_trace(["vibration", "acc"])
            if thrust_stats:
                bullets.append(f"• Thrust climbs to about {thrust_stats['max_y']:.0f} gf near {_fmt_x_val(thrust_stats['x_at_max'])}.")  # type: ignore[arg-type]
            if vib_stats:
                bullets.append(f"• Vibration ({vib_name}) is highest above {_fmt_x_val(vib_stats['x_at_max'])}, indicating bands to check for structural comfort.")  # type: ignore[arg-type]
            bullets.append("• Comparing these curves highlights throttle regions that deliver strong thrust with acceptable vibration levels.")
        elif "acceleration response" in heading_lower:
            # Look across all acceleration traces and identify the dominant axis.
            if trace_stats:
                best_name = max(trace_stats.items(), key=lambda kv: kv[1]["max_y"])[0]
                best = trace_stats[best_name]
                bullets.append(f"• The largest acceleration excursion occurs on '{best_name}' with a peak of about {best['max_y']:.2f} g.")  # type: ignore[arg-type]
                bullets.append(f"• This peak happens at about {_fmt_x_val(best['x_at_max'])}, highlighting the most dynamic phase of the test run.")  # type: ignore[arg-type]
                bullets.append("• Tracking these components helps verify that the airframe and mounting can tolerate transient loads in the tested envelope.")

        # Generic fallback when we do not recognise the heading.
        if not bullets:
            try:
                layout = getattr(fig_obj, "layout", None)
                xaxis = getattr(layout, "xaxis", None)
                yaxis = getattr(layout, "yaxis", None)
                yaxis2 = getattr(layout, "yaxis2", None)
                x_title = str(getattr(getattr(xaxis, "title", None), "text", "") or "").strip()
                y_title = str(getattr(getattr(yaxis, "title", None), "text", "") or "").strip()
                y2_title = str(getattr(getattr(yaxis2, "title", None), "text", "") or "").strip()
            except Exception:
                x_title = y_title = y2_title = ""

            axes_parts = [t for t in (y_title, y2_title) if t]
            if axes_parts:
                axes_phrase = " and ".join(axes_parts) if len(axes_parts) == 2 else axes_parts[0]
            else:
                axes_phrase = "key performance metrics"
            x_phrase = x_title or "operating conditions"

            bullets.append(f"• This graph shows {axes_phrase} versus {x_phrase} for the current motor test.")
            bullets.append("• It helps relate input conditions to output response so that suitable operating regions can be selected.")

        # Return at most two short paragraphs so that the body text under
        # each summary/performance graph stays concise and does not crowd
        # the page layout.
        return [Paragraph(txt, normal_style) for txt in bullets[:2]]

    # Single body font size for all sections (neat, consistent report visibility)
    body_font_size = 10
    body_leading = 12
    normal_style.fontSize = body_font_size
    normal_style.leading = body_leading
    # Justify all body text for a clean report look
    normal_style.alignment = TA_JUSTIFY

    # Table cell styles: same font size as body; header left-aligned, value cells right-aligned
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=normal_style,
        fontSize=body_font_size,
        leading=body_leading,
        wordWrap="normal",  # wrap long text so it fits in cell
    )
    table_cell_style_header = ParagraphStyle(
        "TableCellHeader", parent=table_cell_style, alignment=TA_LEFT,
    )
    table_cell_style_value = ParagraphStyle(
        "TableCellValue", parent=table_cell_style, alignment=TA_RIGHT,
    )

    # ──────────────────────────────────────────────────────────────
    # COVER PAGE  (full-bleed PNG or fallback text cover)
    # ──────────────────────────────────────────────────────────────

    # Use separate lists so we can do two-pass TOC: cover, then TOC, then content
    cover_flowables = []
    content_flowables = []
    use_toc_markers = bool(include_table_of_contents)

    # Decide how to render the cover:
    # - If a full‑page PNG exists, first page uses the 'cover' template and
    #   we immediately switch to 'portrait' for subsequent pages.
    # - Otherwise, we fall back to a simple text cover on a portrait page.
    include_text_cover = False

    if include_cover_page and use_fullpage_cover and cover_path:
        # Use the dedicated full‑bleed cover template for page 1.
        # Then switch to portrait for the next page.
        cover_flowables.append(NextPageTemplate('portrait'))
        cover_flowables.append(PageBreak())
    elif include_cover_page:
        include_text_cover = True

    if include_text_cover:
        # Fallback simple text cover as a normal flowable page (with margins).
        cover_flowables.append(ConditionalSpacer(1, 60 * mm))
        cover_flowables.append(Paragraph("Motor Data Report", title_style))
        cover_flowables.append(ConditionalSpacer(1, 10 * mm))
        cover_flowables.append(Paragraph("Generated by RotriX Dashboard", normal_style))
        cover_flowables.append(ConditionalSpacer(1, 5 * mm))
        cover_flowables.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
        cover_flowables.append(PageBreak())

    # Pre-compute cleaned file-info text so we know whether a File Info section
    # will actually be rendered. This lets us avoid inserting an extra blank
    # page if the user has turned off "Include File Info" for the report.
    info_text = clean_file_info_text(getattr(st.session_state, "report_file_info_text", ""))

    # Build hierarchical TOC structure (level 1–3, title, marker_id) for two-pass TOC
    graph_entries_toc = getattr(st.session_state, "report_graph_entries", {}) or {}
    try:
        fig_line_toc = getattr(st.session_state, "report_throttle_line_fig", None)
        fig_bar_toc = getattr(st.session_state, "report_throttle_bar_fig", None)
        has_throttle_summary_toc = any(
            [fig_line_toc is not None, fig_bar_toc is not None]
        )
    except Exception:
        has_throttle_summary_toc = False

    toc_structure = []  # list of (level, title, marker_id or None)
    if include_table_of_contents:
        toc_structure.append((1, "MOTOR DATA REPORT", None))
        if include_info:
            toc_structure.append((2, "File Info", "file_info"))
        if "Sorted Performance Table" in selected_table_keys:
            toc_structure.append((2, "Sorted Performance Table", "sorted_table"))
        toc_structure.append((1, "MOTOR DATA ANALYSIS", None))

        if has_throttle_summary_toc:
            toc_structure.append((2, "Summary Graph", None))
            toc_structure.append((3, "Time Spent on Each Throttle Operating Range", "toc_throttle_1"))
            toc_structure.append((3, "Thrust Evolution Over Time Across Throttle Bands", "toc_throttle_2"))
        toc_structure.append((2, "Performance Curves", None))
        for gi, g in enumerate(selected_graph_keys):
            entry_toc = graph_entries_toc.get(g, {})
            title = entry_toc.get("heading") or g
            toc_structure.append((3, title, f"toc_graph_{gi + 1}"))
        toc_structure.append((1, "INSIGHTS", "insights"))

    # Section 1: MOTOR DATA REPORT (heading 1. then 1.1. File Info, 1.2. Sorted Performance Table)
    has_section1 = (include_info and info_text) or (
        "Sorted Performance Table" in (selected_table_keys or []) and getattr(st.session_state, "report_sorted_table_df", None) is not None
    )
    if has_section1:
        content_flowables.append(Paragraph("1. MOTOR DATA REPORT", heading_style))
        content_flowables.append(ConditionalSpacer(1, 2 * mm))
        intro_1 = (
            "This section documents the motor test configuration and core numerical results that form the "
            "basis for later plots. It records the bench setup and device details, then presents the main "
            "performance data so that operating points can be compared across different reports."
        )
        content_flowables.append(Paragraph(intro_1, normal_style))
        content_flowables.append(ConditionalSpacer(1, 4 * mm))

    # File info: one bordered table with full-width title row and two-column parameter rows (as in user report)
    if include_info and info_text:
        if use_toc_markers:
            content_flowables.append(MarkerPlaceholder("file_info"))
        content_flowables.append(Paragraph("1.1. File Info", heading_style))
        file_info_intro = (
            "File information captures the test bench, motor, propeller, ESC, power supply, and operator "
            "details used for this run. These entries document the exact hardware and conditions so that "
            "results can be traced, compared, or repeated in future tests."
        )
        content_flowables.append(Paragraph(file_info_intro, normal_style))
        content_flowables.append(ConditionalSpacer(1, 2 * mm))

        info_rows = parse_file_info_to_table(info_text)
        if info_rows:
            def _is_section_header_row(key: str, value: str) -> bool:
                raw = f"{key or ''} {value or ''}".strip().lower()
                if not raw:
                    return True
                # Remove decorative stars and normalize spaces.
                raw = raw.replace("*", " ")
                raw = " ".join(raw.split())
                return raw in {"test details", "system settings"}

            # Remove section title separators so the PDF shows one clean table.
            filtered_rows = [
                (k or "", v or "")
                for (k, v) in info_rows
                if not _is_section_header_row(k or "", v or "")
            ]
            if not filtered_rows:
                filtered_rows = []

            line_color = HexColor("#dee2e6")
            key_width = doc.width * 0.38
            value_width = doc.width - key_width
            table_data = []

            # Single, two-column table: Field | Value.
            for key, value in filtered_rows:
                if key.strip():
                    table_data.append([Paragraph(key.strip(), normal_style), Paragraph(value.strip(), normal_style)])

            if table_data:
                file_info_table = RLTable(table_data, colWidths=[key_width, value_width])
                style_commands = [
                    ("BOX", (0, 0), (-1, -1), 0.5, line_color),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, line_color),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (0, -1), "LEFT"),
                    ("ALIGN", (1, 0), (1, -1), "LEFT"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
                file_info_table.setStyle(TableStyle(style_commands))
                content_flowables.append(file_info_table)
            content_flowables.append(ConditionalSpacer(1, 4 * mm))

    # Helper: add a DataFrame as a table
    # Uses Paragraph for every cell so long column names and values wrap within cell width.
    # auto_fit=True  -> distribute column width evenly so text wraps to fit
    # auto_fit=False -> same (even distribution) for consistent wrapping
    def _add_df_table(df, title, auto_fit: bool = True):
        if df is None or df.empty:
            return
        # Only add a heading if a non-empty title is provided. This lets
        # callers insert custom paragraphs between the heading and table
        # (e.g. the Sorted Performance Table description).
        if title:
            content_flowables.append(Paragraph(title, heading_style))
        # Format numeric columns similar to HTML
        df_local = df.copy()
        for col in df_local.columns:
            if pd.api.types.is_numeric_dtype(df_local[col]):
                df_local[col] = df_local[col].apply(
                    lambda x: f"{x:.4f}".rstrip("0").rstrip(".") if pd.notna(x) and isinstance(x, float) else x
                )
        col_count = len(df_local.columns)
        if col_count == 0:
            return

        def _cell_para(val, style):
            return Paragraph(html.escape(str(val) if val is not None and pd.notna(val) else ""), style)

        # Header row: centered column names
        header_row = [_cell_para(col, table_cell_style_header) for col in df_local.columns]
        # Data rows: right-aligned values
        data_rows = [[_cell_para(val, table_cell_style_value) for val in row] for row in df_local.values.tolist()]
        data = [header_row] + data_rows

        # Equal column widths so text wraps within each cell and fits the page
        col_widths = [doc.width / col_count] * col_count
        table = RLTable(data, repeatRows=1, colWidths=col_widths)
        # Alignment:
        # - Header row (row 0): centered for a clean, balanced header.
        # - Value cells (rows 1..): right-aligned and vertically centered for numeric readability.
        table_style = TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.black),
                # Center header labels
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                # Right-align all data cells
                ("ALIGN", (0, 1), (-1, -1), "RIGHT"),
                # Vertically center all cells
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 2),
                ("GRID", (0, 0), (-1, -1), 0.25, rl_colors.grey),
            ]
        )
        table.setStyle(table_style)
        content_flowables.append(table)
        content_flowables.append(ConditionalSpacer(1, 6 * mm))

    # Helper: add a graph-only section (no table) with heading and layout similar
    # to how we render \"Graph 1\" when its table is not selected.
    def _add_graph_only_section(title: str, fig_obj):
        if fig_obj is None:
            return
        img_bytes = _fig_to_image_bytes(fig_obj)
        if not img_bytes or len(img_bytes) < 100:
            return
        try:
            # Validate PNG header
            if img_bytes[:8] != b'\x89PNG\r\n\x1a\n':
                return
            rl_img = RLImage(BytesIO(img_bytes))
            # Same max width and height as main report graphs (doc.width, 10 cm)
            max_width = doc.width
            graph_height_mm = 100
            iw, ih = rl_img.drawWidth, rl_img.drawHeight
            if iw <= 0 or ih <= 0:
                return
            scale = min(max_width / iw, (graph_height_mm * mm) / ih)
            rl_img.drawWidth = iw * scale
            rl_img.drawHeight = ih * scale

            block = [
                Paragraph(title, heading_style),
                ConditionalSpacer(1, 4 * mm),
                rl_img,
                ConditionalSpacer(1, 6 * mm),
            ]
            content_flowables.append(KeepTogether(block))
            content_flowables.append(ConditionalSpacer(1, 4 * mm))
        except Exception:
            # If anything goes wrong, skip this graph rather than breaking PDF
            pass

    # Restrict tables included in the PDF:
    # - Keep only "Sorted Performance Table"
    # - Exclude "Raw Data Table", "Summary Statistics", and all per-graph tables
    pdf_table_keys = []
    if "Sorted Performance Table" in selected_table_keys:
        pdf_table_keys.append("Sorted Performance Table")


    # Sorted Performance Table early if selected (PDF-only filtered keys)
    # (Now rendered in portrait like the rest of the document.)
    if "Sorted Performance Table" in pdf_table_keys:
        df_sorted = getattr(st.session_state, "report_sorted_table_df", None)
        if df_sorted is not None:
            # Remove acceleration, motor efficiency, and prop mechanical eff columns for report
            df_sorted = _drop_sorted_table_report_columns(df_sorted)
            if df_sorted is not None and not df_sorted.empty:
                # If File Info is present, start the sorted table on the next page
                # so section 1.2 always begins at the top of a fresh page.
                if include_info and info_text:
                    content_flowables.append(PageBreak())
                # TOC marker should sit on the same page as the heading (after any page break).
                if use_toc_markers:
                    content_flowables.append(MarkerPlaceholder("sorted_table"))
                # Numbering: 1.1. when File Info is missing (matches TOC); 1.2. when File Info is present
                sorted_table_heading = "1.1. Sorted Performance Table" if not (include_info and info_text) else "1.2. Sorted Performance Table"
                # Spacer before the Sorted Performance Table section so it reads like a
                # distinct subsection under "Motor Data Report".
                if include_info and info_text:
                    content_flowables.append(ConditionalSpacer(1, 4 * mm))
                # Section heading for the table
                content_flowables.append(Paragraph(sorted_table_heading, heading_style))
                # Short description placed immediately under the heading, before the table.
                # Describe the table, including the throttle range and interval used for this test if available.
                thr_start = st.session_state.get("single_file_throttle_min_input")
                thr_end = st.session_state.get("single_file_throttle_max_input")
                thr_interval = st.session_state.get("single_file_throttle_interval_input")
                if isinstance(thr_start, (int, float)) and isinstance(thr_end, (int, float)) and isinstance(thr_interval, (int, float)):
                    start_str = f"{thr_start:g}"
                    end_str = f"{thr_end:g}"
                    interval_str = f"{thr_interval:g}"
                    range_phrase = f"from {start_str}% to {end_str}% in {interval_str}% throttle steps "
                else:
                    range_phrase = ""
                sorted_intro = (
                    f"The sorted performance table lists each throttle step {range_phrase}"
                    "with its measured voltage, current, thrust, torque, system efficiency, and electrical power. "
                    "Sorting by thrust makes it easy to identify the operating points that deliver the required load "
                    "while keeping power usage in check."
                )
                content_flowables.append(Paragraph(sorted_intro, normal_style))
                content_flowables.append(ConditionalSpacer(1, 2 * mm))
                # Use short header names in PDF so SystemEffect and Motor Power etc. don't squeeze
                df_sorted_display = df_sorted.rename(columns=lambda c: SORTED_TABLE_PDF_COLUMN_SHORT_NAMES.get(c, c))
                # Table uses repeatRows=1 so column headers repeat on each continuation page.
                # Heading was already added above, so pass an empty title here.
                _add_df_table(df_sorted_display, "", auto_fit=True)
                content_flowables.append(PageBreak())

    # Throttle operating range summary graphs (three charts from Report tab Plot section)
    # New layout:
    #   - All three summary graphs stacked vertically on a single page,
    #     each using the full text width with consistent sizing.
    had_section2_heading = False  # set True when we add "2. MOTOR DATA ANALYSIS" (for graphs-only case)
    try:
        fig_line = getattr(st.session_state, "report_throttle_line_fig", None)
        fig_bar = getattr(st.session_state, "report_throttle_bar_fig", None)
        fig_area = None

        has_any_summary = any([fig_line is not None, fig_bar is not None])

        # Helper to build a full-width summary graph block for this page only.
        # Each block keeps its heading and image together and scales the image
        # so that three graphs can comfortably fit on a single page.
        def _build_summary_block(title: str, fig_obj):
            if fig_obj is None:
                return None
            # For the PDF only, slightly increase legend and tick label font sizes
            # so summary graphs remain readable when compressed onto one page.
            try:
                # Work on the same figure object cached in session_state; by the
                # time we are in PDF generation, the on-screen charts have
                # already been rendered.
                xaxis = getattr(fig_obj.layout, "xaxis", None)
                yaxis = getattr(fig_obj.layout, "yaxis", None)

                def _bump_font(current, default_size):
                    try:
                        size = getattr(current, "size", None)
                    except Exception:
                        size = None
                    base = size if isinstance(size, (int, float)) else default_size
                    return max(base + 2, base)  # increase by ~2pt

                if xaxis is not None:
                    tf = getattr(xaxis, "tickfont", None)
                    new_size = _bump_font(tf, 14)
                    fig_obj.update_layout(
                        xaxis=dict(
                            tickfont=dict(size=new_size, color=getattr(tf, "color", "black")),
                        )
                    )
                if yaxis is not None:
                    tfy = getattr(yaxis, "tickfont", None)
                    new_size_y = _bump_font(tfy, 14)
                    fig_obj.update_layout(
                        yaxis=dict(
                            tickfont=dict(size=new_size_y, color=getattr(tfy, "color", "black")),
                        )
                    )
                legend = getattr(fig_obj.layout, "legend", None)
                if legend is not None:
                    lf = getattr(legend, "font", None)
                    new_legend_size = _bump_font(lf, 12)
                    fig_obj.update_layout(
                        legend=dict(
                            font=dict(size=new_legend_size, color=getattr(lf, "color", "black"))
                        )
                    )
            except Exception:
                # If anything goes wrong while tweaking fonts, fall back to the
                # existing styling without breaking PDF generation.
                pass

            img_bytes = _fig_to_image_bytes(fig_obj)
            if not img_bytes or len(img_bytes) < 100:
                return None
            try:
                if img_bytes[:8] != b'\x89PNG\r\n\x1a\n':
                    return None
                rl_img = RLImage(BytesIO(img_bytes))
                iw, ih = rl_img.drawWidth, rl_img.drawHeight
                if iw <= 0 or ih <= 0:
                    return None
                # Same width and height as main report graphs: doc.width, 10 cm
                max_width = doc.width
                graph_height_mm = 100
                scale = min(max_width / iw, (graph_height_mm * mm) / ih)
                rl_img.drawWidth = iw * scale
                rl_img.drawHeight = ih * scale
                # Same heading style as other section headings for consistent report look,
                # followed by short AI-like insight sentences for this summary graph.
                block = [Paragraph(title, heading_style)]
                for para in _make_graph_insight_paragraphs(title, fig_obj):
                    # Slightly tighter spacing between heading (level 3) and
                    # its bullet insights so the summary section remains compact.
                    block.append(ConditionalSpacer(1, 0.5 * mm))
                    block.append(para)
                block.append(ConditionalSpacer(1, 0.5 * mm))
                block.append(rl_img)
                block.append(ConditionalSpacer(1, 2 * mm))
                return block
            except Exception:
                return None

    # ──────────────────────────────────────────────────────────────
    # SECTION 2: MOTOR DATA ANALYSIS — Summary + Performance Graphs
    # ──────────────────────────────────────────────────────────────

        # Section 2: MOTOR DATA ANALYSIS — 2.1. Summary Graph (only when we have summary figures)
        if has_any_summary:
            had_section2_heading = True
            content_flowables.append(Paragraph("2. MOTOR DATA ANALYSIS", heading_style))
            content_flowables.append(ConditionalSpacer(1, 2 * mm))
            content_flowables.append(Paragraph("2.1. Summary Graph", heading_style))
            summary_intro = (
                "Summary graphs provide a compact view of how the propulsion system behaves across the full "
                "throttle range. They show total time in each band, thrust evolution within bands, and how thrust "
                "(or vibration) stacks over absolute time by throttle region before looking at detailed curves."
            )
            summary_intro_2 = (
                "Together these plots show dwell time, thrust traces per band, and the stacked contribution view "
                "over the test timeline so you can relate duration, peaks, and band-by-band behaviour at a glance."
            )
            content_flowables.append(Paragraph(summary_intro, normal_style))
            content_flowables.append(ConditionalSpacer(1, 1.5 * mm))
            content_flowables.append(Paragraph(summary_intro_2, normal_style))
            # Compact summary metrics table: total runtime, max thrust, max speed, max power
            selected_file_name = st.session_state.get("multi_param_selected_file")
            insights_cache = st.session_state.get("multi_param_file_insights", {})
            metrics = insights_cache.get(selected_file_name or "", {}) if selected_file_name else {}
            runtime_s = metrics.get("runtime_s")
            max_thrust = metrics.get("max_thrust")
            max_rpm = metrics.get("max_rpm")
            max_power = metrics.get("max_power")
            runtime_str = seconds_to_mmss(runtime_s) if isinstance(runtime_s, (int, float)) else "N/A"
            thrust_str = f"{max_thrust:.0f}" if isinstance(max_thrust, (int, float)) else "N/A"
            rpm_str = f"{max_rpm:.0f}" if isinstance(max_rpm, (int, float)) else "N/A"
            power_str = f"{max_power:.0f}" if isinstance(max_power, (int, float)) else "N/A"
            content_flowables.append(ConditionalSpacer(1, 2 * mm))
            # Build a compact 2‑column vertical table:
            #   Column 0: metric label
            #   Column 1: metric value
            metric_rows = [
                ("Total runtime (mm:ss)", runtime_str),
                ("Max thrust (gf)", thrust_str),
                ("Max speed (RPM)", rpm_str),
                ("Max power (W)", power_str),
            ]
            summary_data = [
                [
                    Paragraph(label, table_cell_style_header),
                    Paragraph(value, table_cell_style_value),
                ]
                for label, value in metric_rows
            ]

            summary_col_count = 2
            if summary_col_count > 0:
                # Keep the metrics table visually compact and centred on the page.
                total_table_width = doc.width * 0.4  # ~60% of text width
                summary_col_widths = [total_table_width * 0.55, total_table_width * 0.45]
                summary_table = RLTable(summary_data, colWidths=summary_col_widths, hAlign="CENTER")
                summary_table_style = TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), rl_colors.lightgrey),
                        ("TEXTCOLOR", (0, 0), (0, -1), rl_colors.black),
                        # Left‑align metric labels
                        ("ALIGN", (0, 0), (0, -1), "LEFT"),
                        # Right‑align metric values for numeric readability
                        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        ("GRID", (0, 0), (-1, -1), 0.25, rl_colors.grey),
                    ]
                )
                summary_table.setStyle(summary_table_style)
                content_flowables.append(summary_table)
                content_flowables.append(ConditionalSpacer(1, 4 * mm))

        # Build three full-width summary blocks with numbered headings (2.1.1, 2.1.2, 2.1.3)
        # Order in the report:
        #   2.1.1 – Bar chart: time spent in each throttle range
        #   2.1.2 – Line chart: thrust evolution over time
        bar_block = _build_summary_block(
            "2.1.1. Time Spent on Each Throttle Operating Range",
            fig_bar,
        )
        line_block = _build_summary_block(
            "2.1.2. Thrust Evolution Over Time Across Throttle Bands",
            fig_line,
        )
        # Append in order: bar (2.1.1), line (2.1.2)
        for idx, block in enumerate((bar_block, line_block)):
            if block:
                # Marker inside block so TOC page number is the page where this heading appears
                if use_toc_markers:
                    block = [MarkerPlaceholder(f"toc_throttle_{idx + 1}")] + block
                content_flowables.append(KeepTogether(block))

        if has_any_summary:
            content_flowables.append(PageBreak())
    except Exception:
        # If anything goes wrong in the summary section, skip it without
        # breaking the rest of the PDF generation.
        pass

    # Graphs and their optional tables — 2.2. Performance Curves with 2.2.1, 2.2.2, ...
    graph_entries = getattr(st.session_state, "report_graph_entries", {}) or {}

    # Robustness: if the caller passed an empty list or keys that do not
    # exist in report_graph_entries, fall back to "all graphs we have".
    original_selected_graph_keys = list(selected_graph_keys or [])
    if not selected_graph_keys:
        selected_graph_keys = list(graph_entries.keys())
    else:
        # Keep only keys that currently exist; if this strips everything,
        # also fall back to all graphs so the PDF never omits plotted curves.
        filtered = [k for k in selected_graph_keys if k in graph_entries]
        if filtered:
            selected_graph_keys = filtered
        else:
            selected_graph_keys = list(graph_entries.keys())


    if selected_graph_keys:
        if not had_section2_heading:
            content_flowables.append(Paragraph("2. MOTOR DATA ANALYSIS", heading_style))
            content_flowables.append(ConditionalSpacer(1, 2 * mm))
        content_flowables.append(Paragraph("2.2. Performance Curves", heading_style))
        content_flowables.append(ConditionalSpacer(1, 2 * mm))
        perf_intro = (
            "Performance curves present detailed relationships between throttle, RPM, thrust, torque, "
            "efficiency, power, and acceleration. Each graph focuses on a specific aspect of the propulsion "
            "system so that speed, load, power, vibration, and acceleration behaviour can be reviewed separately."
        )
        perf_intro_2 = (
            "Taken together, these curves help identify operating points that balance thrust, efficiency, power "
            "draw, and dynamic loading, and they provide the evidence needed to choose suitable cruise, climb, "
            "and maximum-performance settings for the tested motor–propeller system."
        )
        content_flowables.append(Paragraph(perf_intro, normal_style))
        content_flowables.append(ConditionalSpacer(1, 1.5 * mm))
        content_flowables.append(Paragraph(perf_intro_2, normal_style))
        content_flowables.append(ConditionalSpacer(1, 4 * mm))
    for gi, key in enumerate(selected_graph_keys):
        entry = graph_entries.get(key)
        if not entry:
            continue
        fig = entry.get("fig")
        # Use the heading set in the Plot tab (e.g. "Speed-Based Performance Trends"); fallback to key
        heading_text = entry.get("heading") or key

        # Numbered subheading: 2.2.1., 2.2.2., ...; marker inside block for correct TOC page
        graph_block = [Paragraph(f"2.2.{gi + 1}. {heading_text}", heading_style)]
        if use_toc_markers:
            graph_block = [MarkerPlaceholder(f"toc_graph_{gi + 1}")] + graph_block

        if fig is not None:
            img_bytes = _fig_to_image_bytes(fig, quality=_quality)

            # Add short AI-style insight sentences for this graph (before the image)
            for para in _make_graph_insight_paragraphs(heading_text, fig):
                # Tighter spacing between level-3 heading and bullet insights.
                graph_block.append(ConditionalSpacer(1, 0.5 * mm))
                graph_block.append(para)


            if img_bytes and len(img_bytes) > 100:  # Ensure we have actual image data (PNG header is ~100 bytes)
                try:
                    # Validate PNG header (starts with \x89PNG\r\n\x1a\n)
                    if img_bytes[:8] != b'\x89PNG\r\n\x1a\n':
                        raise ValueError("Invalid PNG format")
                    
                    # Scale image to fit content width; graph height 10 cm
                    rl_img = RLImage(BytesIO(img_bytes))
                    max_width = doc.width
                    # Height chosen so that two performance graphs, their
                    # headings, and bullet insights still fit on a single
                    # A4 page while keeping the plots easy to read.
                    graph_height_mm = 90  # 9 cm
                    iw, ih = rl_img.drawWidth, rl_img.drawHeight
                    if iw > 0 and ih > 0:  # Ensure valid dimensions
                        scale = min(max_width / iw, (graph_height_mm * mm) / ih)
                        rl_img.drawWidth = iw * scale
                        rl_img.drawHeight = ih * scale

                        # Slightly tighter spacing before the image to keep
                        # the overall block compact with the taller graph.
                        graph_block.append(ConditionalSpacer(1, 1.5 * mm))
                        graph_block.append(rl_img)
                        graph_block.append(ConditionalSpacer(1, 2 * mm))
                    else:
                        raise ValueError(f"Invalid image dimensions: {iw}x{ih}")
                except Exception as img_err:
                    pass
                finally:
                    # Free raw image bytes after embedding in reportlab Image object
                    del img_bytes
                    if _quality and _quality.aggressive_gc:
                        import gc; gc.collect()

        # Add the heading + image block as a single flowable so they stay together
        # Only add if the block has content beyond just the heading
        if len(graph_block) > 1:  # Has more than just the heading
            content_flowables.append(KeepTogether(graph_block))
        elif len(graph_block) == 1:  # Only heading, no image
            # Add just the heading without KeepTogether to avoid issues
            content_flowables.append(graph_block[0])
            content_flowables.append(Paragraph("⚠️ Graph image could not be generated. Please display this graph in the Plot tab first.", normal_style))

        # No extra spacer here; the next section starts with an explicit
        # PageBreak so adding vertical space can cause layout overflows on
        # pages that are already nearly full.

    # Standalone tables (that were not already rendered with graphs).
    # For PDF we only include the Sorted Performance Table, and it has already
    # been rendered above if present in pdf_table_keys, so nothing else to add.
    for table_key in pdf_table_keys:
        if table_key == "Sorted Performance Table":
            # Already rendered above near the header
            continue
        df = None
        if table_key == "Raw Data Table":
            df = getattr(st.session_state, "report_raw_data_df", None)
        elif table_key == "Summary Statistics":
            df = getattr(st.session_state, "summary_stats_df", None)
        elif table_key.endswith(" Table") and table_key.startswith("Graph "):
            graph_name = table_key.replace(" Table", "")
            if graph_name in selected_graph_keys:
                # Already rendered next to the graph
                continue
            entry = graph_entries.get(graph_name)
            if entry:
                df = entry.get("table")
        if df is not None:
            _add_df_table(df, table_key)

    # Section 3: INSIGHTS (summary and key highlights)
    content_flowables.append(PageBreak())
    if use_toc_markers:
        content_flowables.append(MarkerPlaceholder("insights"))
    content_flowables.append(Paragraph("3. INSIGHTS", heading_style))
    content_flowables.append(ConditionalSpacer(1, 2 * mm))
    insights_para = (
        "This section summarises the key findings from the drone motor test in clear, actionable language. "
        "It explains how the motor, propeller, ESC, and power system work together over the tested range, "
        "and highlights the operating regions that are most suitable for reliable thrust, manageable power "
        "draw, and acceptable vibration levels."
    )
    content_flowables.append(Paragraph(insights_para, normal_style))
    content_flowables.append(ConditionalSpacer(1, 4 * mm))
    insights_bold_style = ParagraphStyle(
        "InsightsKey", parent=normal_style, fontName="Helvetica-Bold", fontSize=normal_style.fontSize
    )
    content_flowables.append(Paragraph("Key Highlights:", insights_bold_style))
    content_flowables.append(ConditionalSpacer(1, 2 * mm))
    key_highlights = [
        "Stable and predictable performance across the operating range",
        "Consistent relationship between control inputs and performance outputs",
        "Well-balanced integration of motor, propeller, ESC, and power system",
        "Reliable operation across varying load and operating conditions",
        "Clear identification of efficient and sustainable operating zones",
        "Supports system tuning, mission planning, and performance optimisation",
    ]
    for item in key_highlights:
        content_flowables.append(Paragraph(f"• {item}", normal_style))
        content_flowables.append(ConditionalSpacer(1, 1 * mm))

    # Page drawing callbacks
    def _on_first_page(canvas, doc_obj):
        if include_cover_page and use_fullpage_cover and cover_path:
            try:
                img = ImageReader(cover_path)
                page_width, page_height = A4
                # Draw the image to fill the entire physical page (no margins)
                canvas.drawImage(img, 0, 0, width=page_width, height=page_height)
            except Exception:
                # If anything fails, leave the page blank; content starts from page 2 anyway.
                pass

    def _on_later_pages(canvas, doc_obj):
        # No special headers/footers for now
        pass

    # Build hierarchical TOC flowables (title, dotted leaders, page numbers) for two-pass TOC
    toc_dark_blue = HexColor("#1a365d")
    toc_light_blue = HexColor("#2c5282")

    def make_toc_flowables(toc_entries_with_pages):
        """toc_entries_with_pages: list of (level 1-3, title, page_num). Returns flowables for TOC."""
        if not toc_entries_with_pages:
            return []
        # Assign hierarchical numbers (1, 1.1, 1.2, 2, 2.1, 2.1.1, ...)
        stack = [0, 0, 0]
        numbered = []
        for level, title, page in toc_entries_with_pages:
            stack[level - 1] += 1
            for i in range(level, 3):
                stack[i] = 0
            num_str = ".".join(str(stack[i]) for i in range(level))
            numbered.append((num_str, level, title, page))
        # Styles for TOC
        toc_title_style = ParagraphStyle(
            "TOCTitle2", parent=heading_style, alignment=0, fontSize=16, leading=20, textColor=toc_dark_blue, fontName="Helvetica-Bold"
        )
        toc_l1_style = ParagraphStyle(
            "TOCL1", parent=normal_style, fontSize=11, leading=14, textColor=toc_dark_blue, fontName="Helvetica-Bold",
            leftIndent=0, spaceBefore=6,
        )
        toc_l2_style = ParagraphStyle(
            "TOCL2", parent=normal_style, fontSize=10, leading=12, textColor=toc_light_blue,
            leftIndent=6 * mm, spaceBefore=2,
        )
        toc_l3_style = ParagraphStyle(
            "TOCL3", parent=normal_style, fontSize=10, leading=12, textColor=toc_light_blue,
            leftIndent=12 * mm, spaceBefore=1,
        )
        page_col_width = 22 * mm
        title_col_width = doc.width - page_col_width
        toc_data = []
        for num_str, level, title, page in numbered:
            style = toc_l1_style if level == 1 else (toc_l2_style if level == 2 else toc_l3_style)
            left_cell = Paragraph(f"{num_str}. {title}", style)
            right_cell = TocPageNumber(page, font_size=normal_style.fontSize)
            toc_data.append([left_cell, right_cell])
        toc_table = RLTable(toc_data, colWidths=[title_col_width, page_col_width])
        toc_table.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (0, -1), 8),
            ("RIGHTPADDING", (1, 0), (1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return [
            ConditionalSpacer(1, 8 * mm),
            Paragraph("Table of Contents", toc_title_style),
            ConditionalSpacer(1, 6 * mm),
            toc_table,
            ConditionalSpacer(1, 10 * mm),
            PageBreak(),
        ]

    try:
        # Validate that we have content to build
        if not cover_flowables and not content_flowables:
            st.warning("⚠️ No content available to generate PDF. Please ensure graphs and tables are created first.")
            return None

        if include_table_of_contents and toc_structure:
            # Two-pass: first pass to collect page numbers, second pass with hierarchical TOC
            from reportlab.pdfgen import canvas as rl_canvas
            toc_pages_dict.clear()

            def _replace_toc_markers(flowables):
                """Replace MarkerPlaceholder with TOCMarker (recursively inside KeepTogether)."""
                out = []
                for f in flowables:
                    if isinstance(f, MarkerPlaceholder):
                        out.append(TOCMarker(f.marker_id))
                    elif isinstance(f, KeepTogether) and hasattr(f, "_content"):
                        out.append(KeepTogether(_replace_toc_markers(list(f._content))))
                    else:
                        out.append(f)
                return out

            def _strip_toc_markers(flowables):
                """Remove MarkerPlaceholder from flowables (recursively) for final build."""
                out = []
                for f in flowables:
                    if isinstance(f, MarkerPlaceholder):
                        continue
                    if isinstance(f, KeepTogether) and hasattr(f, "_content"):
                        cleaned = _strip_toc_markers(list(f._content))
                        if cleaned:
                            out.append(KeepTogether(cleaned))
                    else:
                        out.append(f)
                return out

            content_with_markers = _replace_toc_markers(content_flowables)

            if _skip_toc_pass1:
                # Low-RAM mode: skip full first-pass build. Assign sequential page numbers.
                # This saves ~50% peak RAM by not building a throwaway PDF.
                pages = [None] * len(toc_structure)
                _page_counter = 2  # cover is page 1, TOC starts on page 2
                for i, (level, title, marker_id) in enumerate(toc_structure):
                    pages[i] = _page_counter
                    if level == 1 or level == 2:
                        _page_counter += 1
                for i in range(len(pages) - 1, -1, -1):
                    if pages[i] is None and i + 1 < len(pages):
                        pages[i] = pages[i + 1]
            else:
                # Normal mode: full first-pass build to get accurate page numbers
                buffer1 = BytesIO()
                doc1 = BaseDocTemplate(buffer1, pageTemplates=page_templates)
                doc1.build(cover_flowables + content_with_markers)
                # Free pass-1 buffer immediately
                del doc1
                buffer1.close()
                del buffer1
                if force_gc:
                    force_gc()
                else:
                    import gc; gc.collect()

                pages = [None] * len(toc_structure)
                for i, (level, title, marker_id) in enumerate(toc_structure):
                    if marker_id and marker_id in toc_pages_dict:
                        pages[i] = toc_pages_dict[marker_id]
                for i in range(len(pages) - 1, -1, -1):
                    if pages[i] is None and i + 1 < len(pages):
                        pages[i] = pages[i + 1]

            toc_entries_with_pages = [
                (toc_structure[i][0], toc_structure[i][1], pages[i] if pages[i] is not None else 1)
                for i in range(len(toc_structure))
            ]
            # Measure how many pages the TOC will take
            page_count = [0]
            class CountCanvas(rl_canvas.Canvas):
                def showPage(self):
                    page_count[0] += 1
                    rl_canvas.Canvas.showPage(self)
            toc_flowables_dummy = make_toc_flowables([(a, b, 1) for a, b, _ in toc_entries_with_pages])
            buffer_measure = BytesIO()
            doc_measure = BaseDocTemplate(buffer_measure, pageTemplates=page_templates)
            doc_measure.build(cover_flowables + toc_flowables_dummy, canvasmaker=CountCanvas)
            has_cover = bool(include_cover_page and use_fullpage_cover and cover_path)
            num_toc_pages = page_count[0] - (1 if has_cover else 0)
            if num_toc_pages < 1:
                num_toc_pages = 1
            # Adjust pages: add num_toc_pages to every entry (content shifts down)
            toc_final = [
                (lev, ttl, (p + num_toc_pages) if p is not None else (1 + num_toc_pages))
                for lev, ttl, p in toc_entries_with_pages
            ]
            toc_flowables = make_toc_flowables(toc_final)
            content_no_markers = _strip_toc_markers(content_flowables)
            # Cleanup measurement doc/buffer to release memory
            try:
                del doc_measure
            except Exception:
                pass
            try:
                buffer_measure.close()
            except Exception:
                pass
            try:
                del buffer_measure
            except Exception:
                pass
            if force_gc:
                try:
                    force_gc()
                except Exception:
                    pass
            story = cover_flowables + toc_flowables + content_no_markers
        else:
            story = cover_flowables + content_flowables

        # NOTE:
        # BaseDocTemplate.build does not accept onFirstPage / onLaterPages kwargs
        # like SimpleDocTemplate does, so we just call doc.build(story).
        doc.build(story)
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()
        try:
            buffer.close()
        except Exception:
            pass
        return pdf_bytes
    except Exception as e:
        # Silently return empty bytes — caller shows appropriate UI message.
        return b""
