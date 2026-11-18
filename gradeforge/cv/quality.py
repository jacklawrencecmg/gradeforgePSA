"""Image-quality gating: decides which parts of a card can be inspected at all."""
from __future__ import annotations

import cv2
import numpy as np

MIN_PX_ANY = 600
MIN_PX_FINE = 900
GOOD_PX = 1200


def assess_quality(det, rect: np.ndarray, mask: np.ndarray, centering: dict, glare: dict) -> dict:
    issues, requests = [], []
    wpx = det.card_width_px
    res = "insufficient" if wpx < MIN_PX_ANY else "limited" if wpx < GOOD_PX else "good"
    if res == "insufficient":
        issues.append(f"Card is only about {wpx:.0f} px wide in the photo.")
        requests.append(f"Retake with the card at least {GOOD_PX} px wide (currently about {wpx:.0f}).")
    elif res == "limited":
        issues.append(f"Card is about {wpx:.0f} px wide; fine defects (hairline scratches, light whitening) may be missed.")
        requests.append(f"For best results use a photo with the card at least {GOOD_PX} px wide.")

    scale = 1000.0 / rect.shape[1]
    small = cv2.resize(rect, (1000, int(round(rect.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    m_small = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    m_small = cv2.erode(m_small, np.ones((15, 15), np.uint8))
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharp = float(lap[m_small > 0].var()) if (m_small > 0).any() else 0.0
    if sharp < 20.0:
        issues.append("Image looks soft or out of focus (low fine-detail energy).")
        requests.append("Refocus and retake; use a tripod or steady surface.")

    quad = det.quad
    top = np.linalg.norm(quad[1] - quad[0]); bot = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0]); right = np.linalg.norm(quad[2] - quad[1])
    skew = max(abs(top - bot) / max(top, bot), abs(left - right) / max(left, right))
    if skew > 0.08:
        issues.append(f"Camera angle is oblique ({skew * 100:.0f}% edge-length difference); measurements lose accuracy.")
        requests.append("Shoot square-on to the card, with the camera parallel to the surface.")

    if glare["border_overexposed"]:
        issues.append("Borders are overexposed; whitening and surface marks are not separable.")
        requests.append("Reduce exposure or lighting intensity.")
    elif glare["clipped_fraction"] > 0.04:
        issues.append(f"{glare['clipped_fraction'] * 100:.1f}% of the card is blown out (glare or overexposure).")
        requests.append("Retake with diffuse light, or add a second photo at a different light angle.")
    for w in det.warnings:
        issues.append(w)

    cen_ok = bool(centering.get("measurable")) and res != "insufficient" and skew <= 0.15
    return {
        "card_width_px": round(wpx), "resolution_class": res, "sharpness_var": round(sharp, 1),
        "perspective_skew": round(float(skew), 3), "issues": issues, "requests": requests,
        "sufficient_for": {
            "centering": cen_ok,
            "corners": wpx >= MIN_PX_FINE,
            "edges": wpx >= MIN_PX_FINE,
            "ring_surface": wpx >= MIN_PX_FINE and not glare["border_overexposed"],
        },
    }
