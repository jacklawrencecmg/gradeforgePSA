"""PDF report generation (reportlab)."""
from __future__ import annotations

import io
from pathlib import Path

import cv2
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SEV = {"ok": (60, 170, 80), "warn": (230, 160, 30), "bad": (210, 50, 50), "unknown": (130, 130, 130),
       "info": (60, 120, 220)}


def _annotated_png(path: Path, overlays: list[dict], max_w: int = 700) -> bytes | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    for o in overlays:
        if o.get("kind") == "line":
            cv2.line(img, (int(o["x1"]), int(o["y1"])), (int(o["x2"]), int(o["y2"])), (255, 200, 0), 2)
        else:
            rgb = SEV.get(o.get("severity", "info"), SEV["info"])
            cv2.rectangle(img, (int(o["x"]), int(o["y"])), (int(o["x"] + o["w"]), int(o["y"] + o["h"])),
                          (rgb[2], rgb[1], rgb[0]), 3)
    s = max_w / img.shape[1]
    img = cv2.resize(img, (max_w, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else None


def build_pdf(record: dict, image_paths: dict, evidence_dir: Path | None = None) -> bytes:
    styles = getSampleStyleSheet()
    h1, h2, body = styles["Heading1"], styles["Heading2"], styles["BodyText"]
    small = styles["BodyText"].clone("small", fontSize=8, leading=10)
    res = record["result"]
    a = res["assessment"]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch, title="GradeForge report")
    story = [Paragraph("GradeForge inspection report", h1),
             Paragraph(f"Card: {record.get('label') or record['id']} &nbsp; | &nbsp; Analysis ID: {record['id']}", body),
             Paragraph(f"Model: {a['model_version']} &nbsp; | &nbsp; PSA criteria database v{a['standards']['database_version']} "
                       f"({'verified against official page' if a['standards']['official_page_verified'] else 'NOT verified against the official PSA page'})",
                       small), Spacer(1, 8)]

    g = res.get("gem")
    if g:
        from xml.sax.saxutils import escape as _e
        e, ml = g["psa10_eligibility"], g["most_likely"]
        story += [Paragraph("Gem Mint analysis", h2),
                  Paragraph(f"<b>PSA 10 eligibility: {_e(e['label'])}</b>", body), Paragraph(_e(e["summary"]), body),
                  Paragraph(f"Most likely grade: {ml['grade'] if ml['grade'] is not None else 'not given'}"
                            f" &nbsp;|&nbsp; ceiling {g['grade_ceiling']['conservative']} (conservative) / "
                            f"{g['grade_ceiling']['optimistic']} (optimistic)", body)]
        if ml.get("note"):
            story.append(Paragraph(_e(ml["note"]), small))
        story.append(Paragraph(_e(g["dominance"]["rule"]), small))
        if g["reasons_psa9"]:
            story.append(Paragraph("Reasons this card may receive a PSA 9 or lower", body))
            for r in g["reasons_psa9"]:
                story.append(Paragraph(f"• {_e(r['text'])} [{r['strength'].replace('_', ' ')}; {r['source']}]", small))
        if g["reasons_psa10_plausible"]:
            story.append(Paragraph("Evidence keeping a PSA 10 plausible", body))
            for r in g["reasons_psa10_plausible"]:
                story.append(Paragraph(f"• {_e(r['text'])} [{r['source']}]" + (f" Caveat: {_e(r['caveat'])}" if r.get("caveat") else ""), small))
        if g["additional_inspection"]:
            story.append(Paragraph("Additional inspection required", body))
            for q in g["additional_inspection"]:
                story.append(Paragraph(f"• [{q['type']}] <b>{_e(q['what'])}.</b> {_e(q['how'])}", small))
        rows = [[Paragraph(_e(c["label"]), small), Paragraph(_e(c["observed"]), small),
                 Paragraph(c["status"].replace("_", " "), small), Paragraph(c["basis"], small)] for c in g["characteristics"]]
        ct = Table([["Characteristic", "Observed", "Status", "Basis"]] + rows, colWidths=[1.5 * inch, 3.3 * inch, 1.1 * inch, 0.9 * inch])
        ct.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                                ("FONTSIZE", (0, 0), (-1, 0), 8), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story += [Spacer(1, 4), ct, Spacer(1, 4)]
        sd = g["standards"]
        story.append(Paragraph("Official PSA wording used (read from psacard.com/gradingstandards)", body))
        for i in sd["official_items"]:
            story.append(Paragraph(f"• {_e(i['text'])}", small))
        story.append(Paragraph("GradeForge assumptions (not PSA rules, uncalibrated)", body))
        for i in sd["model_assumptions"]:
            story.append(Paragraph(f"• {_e(i)}", small))
        story.append(Paragraph("<b>Prediction reliability:</b> " + _e(g["reliability"]["statement"]), small))
        story.append(Spacer(1, 8))

    if a["prediction_status"] != "ceilings_only":
        story += [Paragraph("Result: insufficient evidence", h2),
                  Paragraph("A grade prediction was not forced. Reasons:", body)]
        for r in a.get("reasons", []):
            story.append(Paragraph(f"• {r}", body))
        story.append(Paragraph("Needed to proceed:", body))
        for r in a.get("requests", []):
            story.append(Paragraph(f"• {r}", body))
    else:
        p10 = a["psa10_eligibility"]
        story += [Paragraph("Assessment", h2)]
        t = Table([["Conservative ceiling", a["conservative_ceiling"]], ["Optimistic ceiling", a["optimistic_ceiling"]],
                   ["Most likely grade", "not provided (no calibrated model)"],
                   ["PSA 10 status", p10["status"].replace("_", " ")],
                   ["Evidence level", f"{a['evidence_level']} ({a['inspection_coverage']['assessed']}/"
                                      f"{a['inspection_coverage']['total']} checklist items assessed)"]],
                  colWidths=[1.8 * inch, 4.9 * inch])
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke)]))
        story += [t, Spacer(1, 6),
                  Paragraph("Ceilings are upper bounds from measured criteria. Photos cannot confirm the absence of every flaw, "
                            "so a lower grade is always possible. No probabilities are reported.", small),
                  Paragraph(p10["detail"], body), Paragraph("PSA 9 versus PSA 10", h2),
                  Paragraph(a["nine_vs_ten"]["summary"], body)]
        rows = [["Criterion", "Observed", "Status"]]
        for c in a["checklist"]:
            rows.append([Paragraph(c["requirement"], small), Paragraph(c["observed"], small),
                         Paragraph(c["status"].replace("_", " "), small)])
        ct = Table(rows, colWidths=[3.0 * inch, 2.4 * inch, 1.3 * inch], repeatRows=1)
        ct.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey), ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke)]))
        story.append(ct)
        if a["grade_limiting"]:
            story.append(Paragraph("Grade-limiting findings", h2))
            for g in a["grade_limiting"]:
                story.append(Paragraph(f"• {g['requirement']}: ceiling {g['conservative_cap']}–{g['optimistic_cap']}. {g['detail']}", body))
        if a.get("requests"):
            story.append(Paragraph("Better images would help", h2))
            for r in a["requests"]:
                story.append(Paragraph(f"• {r}", body))

    for side in ("front", "back"):
        s = res.get(side)
        if not (s and s.get("ok")):
            continue
        story.append(PageBreak())
        story.append(Paragraph(f"{side.capitalize()} inspection", h2))
        p = image_paths.get(side)
        if p:
            png = _annotated_png(p, s["overlays"])
            if png:
                story.append(Image(io.BytesIO(png), width=3.6 * inch, height=3.6 * inch * 1.4))
        c = s["centering"]
        if c.get("measurable"):
            story.append(Paragraph(f"Centering: left/right {c['lr_split'][0]}/{c['lr_split'][1]}, top/bottom "
                                   f"{c['tb_split'][0]}/{c['tb_split'][1]}; worst axis {c['worst_axis']} at {c['worst_pct']}% "
                                   f"(±{c['uncertainty_pct']}).", body))
        else:
            story.append(Paragraph(f"Centering not measurable: {c.get('reason')}", body))
        for issue in s["quality"]["issues"]:
            story.append(Paragraph(f"Image quality: {issue}", small))
        comp = s["completeness"]
        story.append(Paragraph(f"Inspection completeness: {comp['level']}", body))
        for r in comp["rois"]:
            if r["inspectable"] != "yes":
                story.append(Paragraph(f"• {r['label']}: {r['inspectable']}. " + " ".join(r["reasons"] + r["requests"]), small))
        story.append(Paragraph(s["resolution"]["note"], small))
        shown = [f for f in s["findings"] if f["status"] != "rejected" and f["category"] != "glare_region"]
        if shown:
            story.append(Paragraph(f"{side.capitalize()} findings ({len(shown)})", h2))
        for f in shown[:40]:
            state = "CONFIRMED" if f["status"] == "confirmed" else "unconfirmed candidate"
            story.append(Paragraph(f"<b>{f['id']}</b> {f['label']} ({f['target'].replace('_', ' ')}, {f['severity']}) — {state}", body))
            story.append(Paragraph(f["summary"].replace("²", "2"), small))
            story.append(Paragraph(f"Classification: {f['classification_text']}. " + " ".join(f["classification_reasons"]), small))
            det = f["detector"]
            story.append(Paragraph(f"Detector {det['id']} v{det['version']} (classical CV, not a trained model, uncalibrated)."
                                   + (f" Reviewed by {f['review']['reviewer']}: {f['review']['note']}" if f.get("review") else ""), small))
            if evidence_dir is not None:
                imgs = []
                for key in ("primary_annotated", "primary_original") + tuple(k for k in f["evidence"].get("files", {}) if k.startswith(("light_", "macro_"))):
                    rec = f["evidence"].get("files", {}).get(key)
                    if not rec:
                        continue
                    p = evidence_dir / rec["file"]
                    im = cv2.imread(str(p)) if p.is_file() else None
                    if im is None:
                        continue
                    sc = 120.0 / max(im.shape[:2])
                    im = cv2.resize(im, (max(1, int(im.shape[1] * sc)), max(1, int(im.shape[0] * sc))), interpolation=cv2.INTER_AREA)
                    ok, b = cv2.imencode(".png", im)
                    if ok:
                        imgs.append(Image(io.BytesIO(b.tobytes()), width=im.shape[1] * 0.75, height=im.shape[0] * 0.75))
                if imgs:
                    story.append(Table([imgs[:5]]))
            story.append(Spacer(1, 6))

    story += [PageBreak(), Paragraph("Limitations", h2)]
    for text in [
        "PSA's official grading standards page could not be read automatically; numeric limits come from secondary sources and are marked unverified in the criteria database.",
        "Defect thresholds are GradeForge model assumptions, uncalibrated until validated against real PSA outcomes.",
        "Every detector is a hand-written classical image-processing rule; no trained or pretrained defect model is used, and none has been validated on real cards.",
        "Artwork relief is judged only from differences between registered lighting photos. Print lines, staining, foil damage, and alteration/authenticity are not assessed automatically.",
        "Features narrower than about two source-photo pixels cannot be resolved and are never reported as physical damage.",
        "Macro photographs are registered and shown as evidence crops; they are not scored automatically.",
        "This report is a pre-grading aid, not a PSA opinion.",
    ]:
        story.append(Paragraph(f"• {text}", body))
    doc.build(story)
    return buf.getvalue()
