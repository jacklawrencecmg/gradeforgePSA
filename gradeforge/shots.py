"""Photo categories: what each upload slot is for, how to take it, and what GradeForge did with each photo.

Every photo belongs to a named category (header) so the app knows what it is looking at and can tell you exactly which
header a missing photo goes under. Corner and edge close-ups are registered to the card; if a close-up lands in the wrong
place (e.g. uploaded under "top-left corner" but it shows the bottom edge) the app says so.
"""
from __future__ import annotations

CORNER_NAMES = {"tl": "top-left", "tr": "top-right", "br": "bottom-right", "bl": "bottom-left"}
EDGE_NAMES = {"top": "top edge", "right": "right edge", "bottom": "bottom edge", "left": "left edge"}
CORNER_SLOTS = [f"corner_{k}" for k in CORNER_NAMES]
EDGE_SLOTS = [f"edge_{k}" for k in EDGE_NAMES]
CLOSEUP_SLOTS = CORNER_SLOTS + EDGE_SLOTS + ["surface"]
SURFACE_MAX = 4

HEADERS = {
    "overall": "Whole card",
    "angle": "Angled-light photos",
    "corner": "Corner close-ups",
    "edge": "Edge close-ups",
    "surface": "Surface close-ups",
}

HOWTO = {
    "overall": "Lay the card flat on a plain, dark or contrasting surface. Hold the camera directly above it, parallel to the card, with the whole card and a little background "
               "visible on every side. Use soft, even light (a window on a cloudy day or two lamps with diffusers). No glare, no shadows across the card, "
               "and take it out of any sleeve or top-loader. The card should be at least 1200 pixels wide (any modern phone at normal zoom is enough).",
    "angle": "Keep the card and camera exactly where they were for the whole-card photo. Use ONE lamp and take extra photos with the lamp low (about 15-30 degrees above the card) "
             "from the left, then from the right. These make scratches, dents and gloss changes appear as light and shadow. Upload 2-4 of them. Do not move the card between shots.",
    "corner": "One photo per corner. Fill the frame with that corner, but keep about a third of the card's width visible along each edge and include some artwork next to the "
              "corner (the app needs a little pattern to line the close-up up with the card). Camera parallel to the card, in sharp focus, soft light, dark background.",
    "edge": "One photo per edge. Keep the whole length of that edge in the frame with the card edge near the middle of the picture, in sharp focus. "
            "Light from the opposite side shows whitening and nicks best. Include a little artwork next to the edge.",
    "surface": "Use these for anything specific you want checked: a suspected scratch, dent, print line, stain or dull patch. Fill about a third of the frame with the area, "
               "include some surrounding artwork so the app can place it on the card, and tilt the light so the flaw casts a shadow or glint.",
}

WHY = {
    "overall": "Centering, overall condition, and where everything is on the card. Required.",
    "angle": "The only way GradeForge can look for scratches, dents, creases and gloss loss inside the artwork.",
    "corner": "Close look at whitening, softness and chips. PSA 10 requires four sharp corners.",
    "edge": "Close look at edge whitening, nicks and rough cuts.",
    "surface": "A closer look at a specific suspected flaw, shown next to the finding it covers.",
}


def slot_label(slot: str) -> str:
    if slot.startswith("corner_"):
        return f"{CORNER_NAMES[slot[7:]].capitalize()} corner"
    if slot.startswith("edge_"):
        return EDGE_NAMES[slot[5:]].capitalize()
    return "Surface close-up"


def slot_header(slot: str) -> str:
    return HEADERS["corner" if slot.startswith("corner_") else "edge" if slot.startswith("edge_") else "surface"]


def upload_path(side: str, slot: str) -> str:
    """Human wording of where in the upload form a photo goes."""
    return f"{side.capitalize()} → {slot_header(slot)} → {slot_label(slot)}"


def check_slot(slot: str, footprint: list[float], w: int, h: int) -> dict:
    """Does a registered close-up really show the area it was uploaded under? footprint = [x0,y0,x1,y1] on the rectified canvas."""
    if slot == "surface" or not footprint:
        return {"ok": True, "note": ""}
    x0, y0, x1, y1 = footprint
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    area = max(0.0, (x1 - x0) * (y1 - y0)) / float(w * h)
    if area > 0.6:
        return {"ok": False, "note": "This photo shows most of the card, not a close-up. It was not counted as a close-up of that spot."}
    if slot.startswith("corner_"):
        k = slot[7:]
        want_x = cx < 0.42 * w if k in ("tl", "bl") else cx > 0.58 * w
        want_y = cy < 0.42 * h if k in ("tl", "tr") else cy > 0.58 * h
        ok = want_x and want_y
    else:
        k = slot[5:]
        ok = {"top": cy < 0.3 * h, "bottom": cy > 0.7 * h, "left": cx < 0.3 * w, "right": cx > 0.7 * w}[k]
    return {"ok": bool(ok), "note": "" if ok else f"This photo appears to show a different part of the card than the {slot_label(slot).lower()}. "
                                                   "Check which photo you chose, or retake it so that spot is in the middle of the frame."}


# ---------------------------------------------------------------------------------------------------------------
def build_inventory(result: dict, uploads: dict | None) -> dict:
    """Every uploaded photo, under its category header, with what GradeForge did with it. Also lists what is missing."""
    uploads = uploads or {}
    out = {"sides": {}, "totals": {"uploaded": 0, "used": 0, "not_used": 0}}
    for side in ("front", "back"):
        s = result.get(side)
        up = uploads.get(side, {})
        cats = []

        def add(cid, status, items, missing=None, extra=None):
            cats.append({"id": cid, "header": HEADERS[cid], "purpose": WHY[cid], "howto": HOWTO[cid], "status": status, "items": items,
                         "missing": missing or [], **(extra or {})})

        # whole card
        if up.get("primary") is None and not (s and s.get("ok")):
            add("overall", "missing", [], [f"{side.capitalize()} → Whole card"])
        else:
            ok = bool(s and s.get("ok"))
            item = {"name": up.get("primary") or "uploaded photo", "used": ok,
                    "note": ("Card found and measured." if ok else (s or {}).get("reason", "The card could not be found in this photo."))}
            if ok:
                q = s["quality"]
                if not all(q["sufficient_for"].values()):
                    item["note"] += " Resolution or exposure is low for: " + ", ".join(k for k, v in q["sufficient_for"].items() if not v) + "."
            add("overall", "provided" if ok else "problem", [item])

        if not (s and s.get("ok")):
            for cid in ("angle", "corner", "edge", "surface"):
                add(cid, "missing", [])
            out["sides"][side] = cats
            continue

        # angled-light photos
        reg = s["registration"]["lighting"]
        stack = s["registration"]["lighting_stack"]
        names = up.get("lights", [])
        items = [{"name": names[i] if i < len(names) else r["role"], "used": bool(r["ok"]),
                  "note": "Lined up with the main photo." if r["ok"] else "Not used: " + (r.get("reason") or "could not be lined up with the main photo.")}
                 for i, r in enumerate(reg)]
        if items and stack["images"] >= 2 and not stack["diversity_ok"]:
            for it in items:
                it["note"] += " The lighting looks too similar to the main photo to reveal relief; move the lamp further to one side or lower."
        status = "missing" if not items else ("provided" if sum(i["used"] for i in items) >= 1 and stack["diversity_ok"] else "problem")
        add("angle", status, items, [] if status == "provided" else [f"{side.capitalize()} → Angled-light photos (at least 2)"],
            {"count_used": sum(i["used"] for i in items)})

        # close-ups: map by slot
        macros = s["registration"].get("macros", [])
        w, h = s["detection"]["canvas"]
        by_slot: dict[str, list] = {}
        for m in macros:
            by_slot.setdefault(m.get("slot") or "surface", []).append(m)

        def macro_item(m, slot):
            if not m.get("registered"):
                return {"name": m.get("name"), "used": False, "slot": slot,
                        "note": "Not used: " + (m.get("reason") or "could not be matched to the card") +
                                ". Retake with a little more of the card (including some artwork) in the frame."}
            chk = check_slot(slot, m["footprint"], w, h)
            return {"name": m.get("name"), "used": bool(chk["ok"]), "slot": slot,
                    "note": "Matched to the card and shown next to findings in that area." if chk["ok"] else "Not counted: " + chk["note"]}

        for cid, slots in (("corner", CORNER_SLOTS), ("edge", EDGE_SLOTS)):
            items, missing, any_problem = [], [], False
            for sl in slots:
                got = by_slot.get(sl, [])
                if not got:
                    missing.append(upload_path(side, sl))
                    items.append({"name": None, "used": False, "slot": sl, "label": slot_label(sl), "note": "Not provided."})
                    continue
                for m in got:
                    it = macro_item(m, sl)
                    it["label"] = slot_label(sl)
                    items.append(it)
                    any_problem |= not it["used"]
            n_have = sum(1 for i in items if i["name"])
            status = "missing" if n_have == 0 else ("problem" if any_problem else "partial" if missing else "provided")
            add(cid, status, items, missing)
        sur = [macro_item(m, "surface") | {"label": "Surface close-up"} for m in by_slot.get("surface", [])]
        add("surface", "provided" if sur and all(i["used"] for i in sur) else "problem" if sur else "optional", sur, [])
        out["sides"][side] = cats

    for cats in out["sides"].values():
        for c in cats:
            for it in c["items"]:
                if it.get("name"):
                    out["totals"]["uploaded"] += 1
                    out["totals"]["used" if it["used"] else "not_used"] += 1
    return out
