"""Hand-built pipeline results for fast, deterministic tests of the Gem Mint engine (no image processing)."""
from gradeforge.findings import assign_ids, new_finding

CORNERS = ("tl", "tr", "br", "bl")
EDGES = ("top", "right", "bottom", "left")


def rois(relief="yes", border="yes", corners="yes", edges="yes", centering="yes"):
    out = []
    for c in CORNERS:
        out.append({"id": f"corner_{c}", "label": f"Corner {c.upper()}", "inspectable": corners, "reasons": [], "requests": []})
    for e in EDGES:
        out.append({"id": f"edge_{e}", "label": f"Edge {e}", "inspectable": edges, "reasons": [], "requests": []})
    out.append({"id": "centering", "label": "Centering", "inspectable": centering, "reasons": [], "requests": []})
    out.append({"id": "border_surface", "label": "Border surface", "inspectable": border,
                "reasons": [] if border == "yes" else ["One photo only."], "requests": []})
    out.append({"id": "artwork_relief", "label": "Artwork relief / gloss", "inspectable": relief,
                "reasons": [] if relief == "yes" else ["Needs two lighting photos."], "requests": []})
    out.append({"id": "artwork_print_and_stain", "label": "Artwork print lines, staining, foil damage", "inspectable": "no",
                "reasons": ["Needs a person."], "requests": []})
    return out


def finding(side, category, target, severity="minor", cls="physical_probable_albedo", status="candidate", **meas):
    f = new_finding(side, category, target, [10, 10, 40, 40], {"strength": 10.0, **meas}, severity, cls, [], "test.detector", f"{category} test")
    f["status"] = status
    return f


def side(name="front", pct=51.0, unc=0.8, measurable=True, findings=(), relief="yes", border="yes", lights=3, corners="yes", edges="yes"):
    fs = list(findings)
    assign_ids(fs, name)
    return {
        "side": name, "ok": True,
        "centering": ({"measurable": True, "worst_pct": pct, "uncertainty_pct": unc, "worst_axis": "left-right",
                       "lr_split": [pct, 100 - pct], "tb_split": [50.0, 50.0]} if measurable else {"measurable": False, "reason": "no border"}),
        "quality": {"sufficient_for": {"centering": measurable, "corners": True, "edges": True, "ring_surface": True},
                    "issues": [], "requests": [], "sharpness_var": 120.0},
        "resolution": {"min_resolvable_feature_mm": 0.12},
        "completeness": {"rois": rois(relief, border, corners, edges, "yes" if measurable else "no"), "level": "complete"},
        "registration": {"lighting_stack": {"images": lights}},
        "surface": {"glare": {"clipped_fraction": 0.0}, "artwork_relief_assessed": relief == "yes",
                    "ring": {"assessed": True}, "artwork_note": ""},
        "wear": {"corners": [{"id": c, "status": "clean_observed", "whitening_fraction": 0.0, "radius_deviation_mm": 0.0} for c in CORNERS],
                 "edges": [{"id": e, "status": "clean_observed", "longest_whitening_run_mm": 0.0, "roughness_rms_mm": 0.02} for e in EDGES]},
        "findings": fs,
    }


def result(front=None, back=None, finish="standard"):
    return {"front": front if front is not None else side("front"), "back": back if back is not None else side("back", pct=52.0),
            "card_traits": {"finish": finish}}
