"""Milestone 0.7: builds the private interactive desktop prototype comparing
spatial-continuity transition methods between real captured photographs.

Given the manifest/frames produced by spatial_transition_prototype.py plus
the Milestone 0.6 spatial graph, this assembles a single self-contained
index.html (private output; copies photos + rendered frames alongside it)
that lets the founder:

  - pick one of the 2-3 instrumented photo pairs
  - switch between the four transition methods (hard cut / crossfade /
    geometry-aligned camera move / correspondence warp) for that pair and
    play/scrub through it
  - view the landmark-match diagnostic overlay for that pair
  - "continue through the room" afterward: from the destination photo, see
    its other spatially connected neighbors (from the full 11-node graph)
    and hop to any of them via a plain crossfade (the honest default for
    edges that were not specially instrumented this milestone)

This script itself contains no photos, poses, or other private data.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Remember -- Milestone 0.7 spatial transition prototype</title>
<style>
  body { background:#111; color:#eee; font-family: -apple-system, Segoe UI, sans-serif; margin:0; padding:20px; }
  h1 { font-size: 18px; margin: 0 0 4px 0; }
  .sub { color:#999; font-size:13px; margin-bottom:16px; }
  .layout { display:flex; gap:20px; align-items:flex-start; }
  .stage { position:relative; width:720px; height:540px; background:#000; border:1px solid #333; overflow:hidden; }
  .stage img { position:absolute; top:0; left:0; width:100%; height:100%; object-fit:contain; }
  .panel { width:360px; }
  .row { margin-bottom:14px; }
  label { display:block; font-size:12px; color:#aaa; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.05em;}
  select, button { background:#222; color:#eee; border:1px solid #444; padding:6px 10px; border-radius:4px; font-size:13px; cursor:pointer;}
  button.active { background:#3a6; border-color:#3a6; color:#fff; }
  .methodRow button { margin-right:6px; margin-bottom:6px; }
  input[type=range] { width:100%; }
  .hud { background:#1a1a1a; border:1px solid #333; border-radius:6px; padding:10px; font-size:12px; line-height:1.6; }
  .hud .k { color:#8bf; }
  .badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; margin-right:6px; }
  .badge.captured { background:#2a5; }
  .badge.reconstructed { background:#a52; }
  .badge.notcredible { background:#822; }
  .neighborBtn { display:block; width:100%; text-align:left; margin-bottom:6px; }
  .overlayToggle { margin-top:10px; }
  .diagnosticImg { width:100%; border:1px solid #333; margin-top:8px; display:none; }
</style>
</head>
<body>
<h1>Remember -- Milestone 0.7: spatial-continuity transitions</h1>
<div class="sub">Every frame you see originated from one of the two real captured photographs for the selected pair. Nothing is generated.</div>
<div class="layout">
  <div>
    <div class="stage" id="stage"><img id="frameImg" src=""></div>
    <div class="row" style="margin-top:10px;">
      <input type="range" id="scrub" min="0" max="1" value="0" step="1">
    </div>
    <div class="row">
      <button id="playBtn">Play</button>
      <button id="prevFrame">&lt; frame</button>
      <button id="nextFrame">frame &gt;</button>
    </div>
  </div>
  <div class="panel">
    <div class="row">
      <label>Photo pair</label>
      <select id="pairSelect"></select>
    </div>
    <div class="row methodRow">
      <label>Transition method</label>
      <div id="methodButtons"></div>
    </div>
    <div class="row hud" id="hud"></div>
    <div class="row overlayToggle">
      <button id="overlayBtn">Show landmark-match diagnostic</button>
      <img class="diagnosticImg" id="overlayImg">
    </div>
    <div class="row">
      <label>Continue through the room (from destination photo)</label>
      <div id="neighbors"></div>
    </div>
  </div>
</div>
<script>
const DATA = __DATA_JSON__;
const GRAPH = __GRAPH_JSON__;

let currentPairIndex = 0;
let currentMethod = "baseline_hard_cut";
let currentFrame = 0;
let playing = false;
let playTimer = null;

const pairSelect = document.getElementById("pairSelect");
const methodButtons = document.getElementById("methodButtons");
const frameImg = document.getElementById("frameImg");
const scrub = document.getElementById("scrub");
const hud = document.getElementById("hud");
const overlayBtn = document.getElementById("overlayBtn");
const overlayImg = document.getElementById("overlayImg");
const neighborsDiv = document.getElementById("neighbors");

const METHOD_LABELS = {
  baseline_hard_cut: "Baseline A: Hard cut",
  baseline_crossfade: "Baseline B: Crossfade",
  candidate_geometry: "Candidate C: Geometry-aligned move",
  candidate_warp: "Candidate D: Correspondence warp",
};

function framePath(pairIdx, method, frameIdx) {
  const n = frameIdx.toString().padStart(2, "0");
  return `pair_${pairIdx.toString().padStart(2,"0")}/${method}/frame_${n}.png`;
}

function currentPair() {
  return DATA.pairs[currentPairIndex];
}

function buildPairOptions() {
  pairSelect.innerHTML = "";
  DATA.pairs.forEach((p, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${i}: ${p.nameA} <-> ${p.nameB} (landmarks=${p.sharedLandmarksInGraph})`;
    pairSelect.appendChild(opt);
  });
}

function buildMethodButtons() {
  methodButtons.innerHTML = "";
  const pair = currentPair();
  Object.keys(pair.methods).forEach((m) => {
    const btn = document.createElement("button");
    btn.textContent = METHOD_LABELS[m] || m;
    btn.dataset.method = m;
    if (m === currentMethod) btn.classList.add("active");
    btn.onclick = () => { currentMethod = m; currentFrame = 0; refresh(); };
    methodButtons.appendChild(btn);
  });
}

function refresh() {
  const pair = currentPair();
  const methodInfo = pair.methods[currentMethod];
  const frameCount = methodInfo.frameCount;
  scrub.max = frameCount - 1;
  scrub.value = currentFrame;
  frameImg.src = framePath(currentPairIndex, currentMethod, currentFrame);

  [...methodButtons.children].forEach((b) => {
    b.classList.toggle("active", b.dataset.method === currentMethod);
  });

  let credibilityBadge = "";
  if (currentMethod === "candidate_warp") {
    credibilityBadge = methodInfo.credible
      ? `<span class="badge captured">credible warp</span>`
      : `<span class="badge notcredible">not credible -- crossfade fallback shown</span>`;
  }

  const provenance = (currentMethod === "baseline_hard_cut")
    ? `<span class="badge captured">CAPTURED only</span>`
    : `<span class="badge reconstructed">CAPTURED + geometry-repositioned pixels</span>`;

  hud.innerHTML = `
    <div><span class="k">Pair:</span> ${pair.nameA} &harr; ${pair.nameB}</div>
    <div><span class="k">Shared landmarks (graph):</span> ${pair.sharedLandmarksInGraph}</div>
    <div><span class="k">Camera displacement:</span> ${pair.cameraDistance.toFixed(2)} (unscaled SfM units)</div>
    <div><span class="k">Angular change:</span> ${pair.angularDifferenceDegrees.toFixed(1)}&deg;</div>
    <div><span class="k">Geometry consistency:</span> ${pair.geometryConsistency.interpretation}</div>
    <div><span class="k">Feature match:</span> ${pair.featureMatch.nInliers}/${pair.featureMatch.nMatches} inliers (${(pair.featureMatch.inlierRatio*100).toFixed(0)}%)</div>
    <div style="margin-top:6px;">${provenance} ${credibilityBadge}</div>
    <div style="margin-top:6px;"><span class="k">Frame:</span> ${currentFrame + 1} / ${frameCount}</div>
  `;

  overlayImg.src = `pair_${currentPairIndex.toString().padStart(2,"0")}/landmark_overlay.png`;
  buildNeighbors();
}

function buildNeighbors() {
  neighborsDiv.innerHTML = "";
  const pair = currentPair();
  const destName = currentFrame === 0 && currentMethod === "baseline_hard_cut" ? pair.nameA : pair.nameB;
  const neighbors = GRAPH.edges.filter(e => e.from === destName || e.to === destName)
    .map(e => ({ other: e.from === destName ? e.to : e.from, classification: e.classification }))
    .sort((a,b) => (a.classification === "strong" ? -1 : 1));

  const label = document.createElement("div");
  label.style.fontSize = "12px";
  label.style.color = "#999";
  label.textContent = `From ${destName}, spatially connected photos (crossfade hop):`;
  neighborsDiv.appendChild(label);

  neighbors.slice(0, 6).forEach(n => {
    const btn = document.createElement("button");
    btn.className = "neighborBtn";
    btn.textContent = `${n.other} [${n.classification}]`;
    btn.onclick = () => alert("This neighbor is outside the 3 instrumented pairs for Milestone 0.7 -- " +
      "in a full build this would crossfade to " + n.other + ". Only the selected pairs above have " +
      "rendered frames in this prototype.");
    neighborsDiv.appendChild(btn);
  });
}

pairSelect.onchange = () => {
  currentPairIndex = parseInt(pairSelect.value, 10);
  currentMethod = "baseline_hard_cut";
  currentFrame = 0;
  buildMethodButtons();
  refresh();
};

scrub.oninput = () => { currentFrame = parseInt(scrub.value, 10); refresh(); };

document.getElementById("prevFrame").onclick = () => { currentFrame = Math.max(0, currentFrame - 1); refresh(); };
document.getElementById("nextFrame").onclick = () => {
  const max = currentPair().methods[currentMethod].frameCount - 1;
  currentFrame = Math.min(max, currentFrame + 1); refresh();
};

document.getElementById("playBtn").onclick = () => {
  playing = !playing;
  document.getElementById("playBtn").textContent = playing ? "Pause" : "Play";
  if (playing) {
    playTimer = setInterval(() => {
      const max = currentPair().methods[currentMethod].frameCount - 1;
      currentFrame = currentFrame >= max ? 0 : currentFrame + 1;
      refresh();
    }, 90);
  } else {
    clearInterval(playTimer);
  }
};

overlayBtn.onclick = () => {
  const showing = overlayImg.style.display === "block";
  overlayImg.style.display = showing ? "none" : "block";
  overlayBtn.textContent = showing ? "Show landmark-match diagnostic" : "Hide landmark-match diagnostic";
};

buildPairOptions();
buildMethodButtons();
refresh();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", required=True, help="Output dir from spatial_transition_prototype.py (has manifest.json + pair_XX/ frames)")
    parser.add_argument("--graph", required=True, help="Path to private graph.json")
    parser.add_argument("--output", required=True, help="Private output directory for the prototype")
    args = parser.parse_args()

    transitions_dir = Path(args.transitions)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((transitions_dir / "manifest.json").read_text())
    graph = json.loads(Path(args.graph).read_text())
    # Only ship the aggregate edge classifications to the prototype's
    # embedded JSON -- never the poses themselves, even though this whole
    # prototype is a private artifact, to keep the habit consistent.
    graph_public = {"edges": [
        {"from": e["from"], "to": e["to"], "classification": e["classification"]}
        for e in graph["edges"]
    ]}

    # Copy per-pair frame directories + landmark overlays.
    for pair_dir in sorted(transitions_dir.glob("pair_*")):
        dest = out_dir / pair_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(pair_dir, dest)

    html = (HTML_TEMPLATE
            .replace("__DATA_JSON__", json.dumps(manifest))
            .replace("__GRAPH_JSON__", json.dumps(graph_public)))
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote prototype to {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
