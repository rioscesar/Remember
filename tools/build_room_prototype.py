#!/usr/bin/env python3
"""Milestone 0.8, Phases 6-7: minimal continuous-room desktop prototype.

Wraps `structural_completion.py`'s output (a single stitched
ceiling+wall+floor image) in a small pannable HTML viewer so the founder can
look around a *continuous* structural room instead of the isolated-landmark
photo-graph feel of Milestone 0.6, or the slideshow feel of Milestone 0.7.

Two views, toggled by one button:
  - Founder view: `room-structural.png` -- natural colours throughout.
    Real photographic texture is sharp; the inferred wall gaps (filled by
    `evidence_doctrine.bounded_inference_fill`) and the floor/ceiling bands
    are deliberately soft/low-fidelity so they read as "presence, not
    detail" rather than pretending to be captured detail. Critical
    regions (posters, art, the TV) are honestly absent -- rendered as
    plain darkness, never completed.
  - Debug provenance view: `room-provenance.png` -- flat colour-coded
    per-pixel provenance (white=OBSERVED, pale blue=RECONSTRUCTED,
    muted violet=INFERRED, near-black=ABSENT/critical-protected).

This prototype does not attempt true 3D navigation (no new COLMAP/Gaussian
work, per the Milestone 0.8 constraints) -- it is a horizontal pan across
one recovered wall's structural completion, which is the smallest thing
that can honestly demonstrate "continuous structure" instead of "isolated
supported islands."
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Remember -- Milestone 0.8 structural room prototype (private)</title>
<style>
  html, body { margin: 0; height: 100%; background: #0b0b0d; overflow: hidden; font-family: system-ui, sans-serif; }
  #stage { position: relative; width: 100%; height: 100%; overflow: hidden; cursor: grab; }
  #stage.dragging { cursor: grabbing; }
  #room {
    position: absolute; top: 50%; left: 0; height: 140%; transform: translateY(-50%);
    user-select: none; -webkit-user-drag: none; will-change: left;
  }
  #hud { position: fixed; top: 14px; left: 14px; z-index: 10; color: #eee; font-size: 13px; line-height: 1.5; }
  #hud button {
    background: #1c1c22; color: #eee; border: 1px solid #444; border-radius: 6px;
    padding: 8px 14px; cursor: pointer; font-size: 13px;
  }
  #hud button:hover { background: #2a2a33; }
  #caption { margin-top: 8px; opacity: 0.75; max-width: 360px; }
  #private-badge {
    position: fixed; bottom: 14px; right: 14px; color: #a55; font-size: 12px; opacity: 0.7;
  }
</style>
</head>
<body>
<div id="hud">
  <button id="toggle">Switch to debug provenance view</button>
  <div id="caption">Founder view: natural photographic wall, honest low-fidelity floor/ceiling presence, critical content (art/TV/posters) left absent rather than invented. Drag to pan.</div>
</div>
<div id="stage">
  <img id="room" src="room-structural.png" draggable="false" />
</div>
<div id="private-badge">private prototype -- do not distribute source imagery</div>
<script>
  const img = document.getElementById('room');
  const stage = document.getElementById('stage');
  const toggle = document.getElementById('toggle');
  const caption = document.getElementById('caption');
  let founderView = true;
  let dragging = false, startX = 0, startLeft = 0, left = 0;

  function clampLeft(value) {
    const maxLeft = 0;
    const minLeft = stage.clientWidth - img.clientWidth;
    return Math.min(maxLeft, Math.max(minLeft, value));
  }

  stage.addEventListener('pointerdown', (e) => {
    dragging = true; startX = e.clientX; startLeft = left;
    stage.classList.add('dragging');
    stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    left = clampLeft(startLeft + (e.clientX - startX));
    img.style.left = left + 'px';
  });
  stage.addEventListener('pointerup', () => { dragging = false; stage.classList.remove('dragging'); });
  window.addEventListener('resize', () => { left = clampLeft(left); img.style.left = left + 'px'; });

  toggle.addEventListener('click', () => {
    founderView = !founderView;
    img.src = founderView ? 'room-structural.png' : 'room-provenance.png';
    toggle.textContent = founderView ? 'Switch to debug provenance view' : 'Switch to founder view';
    caption.textContent = founderView
      ? 'Founder view: natural photographic wall, honest low-fidelity floor/ceiling presence, critical content (art/TV/posters) left absent rather than invented. Drag to pan.'
      : 'Debug provenance view: white=OBSERVED (single photo), pale blue=RECONSTRUCTED (multiple photos agree), muted violet=INFERRED (bounded conservative fill), near-black=ABSENT (unsupported or critical-protected). Drag to pan.';
  });
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-completion", type=Path, required=True, help="Output dir from structural_completion.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("room-structural.png", "room-provenance.png"):
        shutil.copyfile(args.structural_completion / name, args.output / name)
    (args.output / "index.html").write_text(TEMPLATE, encoding="utf-8")
    print(f"Prototype written to {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
