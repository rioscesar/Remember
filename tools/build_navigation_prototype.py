#!/usr/bin/env python3
"""Milestone 0.6 Phase 6: build the local desktop/browser navigation prototype.

Generates a single self-contained index.html (private output only -- copies
real source photographs and rendered bridge frames alongside it, so this
whole output directory must never be committed) offering three modes for
founder comparison, per the brief:

  A. Chronological -- plain next/previous through the 11 registered photos.
  B. Spatial graph -- click directly between spatially connected photos
     (from spatial_graph.py's edges), no bridging, hard cut.
  C. Hybrid supported spatial memory -- anchors show the real captured
     photograph dominant; the single evidence-graded "strong" edge offers a
     support-gated local radiance cross-dissolve (real reconstructed pixels
     only, faded by the Milestone 0.5 support gate); "weak" edges crossfade
     directly between the two real photographs (no synthesized pixels) and
     the UI biases the user toward the next anchor rather than pretending
     free movement.

This script only assembles pre-rendered material (photos + bridge frames
produced by hybrid_transition_render.py); it renders nothing new itself.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def project_layout(nodes: list[dict]) -> dict[str, tuple[float, float]]:
    centers = np.array([node["center"] for node in nodes])
    centroid = centers.mean(axis=0)
    _, _, axes = np.linalg.svd(centers - centroid, full_matrices=False)
    basis = axes[:2]
    flat = (centers - centroid) @ basis.T
    span = flat.max(axis=0) - flat.min(axis=0)
    span[span < 1e-6] = 1.0
    normalized = (flat - flat.min(axis=0)) / span
    return {node["name"]: (float(normalized[i, 0]), float(1.0 - normalized[i, 1])) for i, node in enumerate(nodes)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--bridge-frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-width", type=int, default=1100)
    args = parser.parse_args()

    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    manifest = json.loads((args.bridge_frames / "manifest.json").read_text(encoding="utf-8"))

    photos_dir = args.output / "photos"
    frames_dir = args.output / "frames"
    photos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    node_names = [node["name"] for node in graph["nodes"]]
    for name in node_names:
        image = cv2.imread(str(args.images / name))
        height, width = image.shape[:2]
        if width > args.max_width:
            new_height = int(height * args.max_width / width)
            image = cv2.resize(image, (args.max_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(photos_dir / name), image, [cv2.IMWRITE_JPEG_QUALITY, 88])

    bridge_frame_lookup = {}
    for edge in manifest["edges"]:
        key = f"{edge['from']}__{edge['to']}"
        source_dir = args.bridge_frames / key
        dest_dir = frames_dir / key
        dest_dir.mkdir(exist_ok=True)
        frame_files = []
        for frame in edge["frames"]:
            src = source_dir / f"frame-{frame['frame']:02d}-gated-rgba.png"
            dst = dest_dir / src.name
            shutil.copyfile(src, dst)
            frame_files.append({"file": f"frames/{key}/{src.name}", **frame})
        bridge_frame_lookup[key] = frame_files

    layout = project_layout(graph["nodes"])

    js_nodes = [{"name": name, "x": layout[name][0], "y": layout[name][1]} for name in node_names]
    js_edges = [
        {
            "from": edge["from"], "to": edge["to"], "classification": edge["classification"],
            "minPathSupport": edge["minPathSupport"], "meanPathSupport": edge["meanPathSupport"],
            "sharedLandmarks": edge["sharedLandmarks"],
            "bridgeFrames": bridge_frame_lookup.get(f"{edge['from']}__{edge['to']}", []),
        }
        for edge in graph["edges"]
    ]

    data_json = json.dumps({"nodes": js_nodes, "edges": js_edges}, indent=None)

    html = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    (args.output / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote navigation prototype to {args.output / 'index.html'}")


HTML_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Remember Milestone 0.6 -- hybrid spatial memory prototype</title>
<style>
  body { margin: 0; background: #0d0d0d; color: #eee; font: 15px/1.4 system-ui; }
  header { padding: 14px 20px; border-bottom: 1px solid #333; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 16px; margin: 0; }
  .modes button { background: #222; color: #eee; border: 1px solid #444; padding: 6px 14px; cursor: pointer; border-radius: 4px; }
  .modes button.active { background: #3a6; border-color: #3a6; color: #032; font-weight: 600; }
  main { display: grid; grid-template-columns: 1.4fr 1fr; gap: 0; height: calc(100vh - 54px); }
  .stage { position: relative; background: #000; overflow: hidden; display: flex; align-items: center; justify-content: center; }
  .stage img.layer { position: absolute; max-width: 100%; max-height: 100%; object-fit: contain; }
  .hud { position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.55); padding: 8px 12px; border-radius: 6px; font-size: 12px; max-width: 320px; }
  .hud b { color: #9f9; }
  .side { border-left: 1px solid #333; padding: 14px; overflow-y: auto; }
  .graph-panel { position: relative; width: 100%; aspect-ratio: 1; background: #141414; border: 1px solid #333; border-radius: 6px; }
  .node-dot { position: absolute; width: 14px; height: 14px; border-radius: 50%; background: #ddd; border: 2px solid #111; transform: translate(-50%, -50%); cursor: pointer; }
  .node-dot.current { background: #6f6; box-shadow: 0 0 8px #6f6; }
  .edge-line { position: absolute; height: 2px; transform-origin: 0 0; background: #777; }
  .edge-line.strong { background: #6c6; height: 3px; }
  .edge-line.weak { background: #66a; opacity: 0.6; }
  .legend { font-size: 12px; margin-top: 10px; color: #aaa; }
  .neighbor-list button { display: block; width: 100%; text-align: left; background: #1c1c1c; color: #eee; border: 1px solid #333; padding: 8px; margin: 4px 0; border-radius: 4px; cursor: pointer; }
  .neighbor-list button:hover { border-color: #6c6; }
  .neighbor-list .tag { font-size: 11px; padding: 1px 6px; border-radius: 3px; margin-left: 6px; }
  .tag.strong { background: #274; }
  .tag.weak { background: #246; }
  .metrics { font-size: 12px; color: #aaa; white-space: pre-wrap; margin-top: 12px; }
</style>
<header>
  <h1>Remember -- Milestone 0.6 navigation prototype</h1>
  <div class="modes">
    <button id="mode-a" onclick="setMode('A')">A: Chronological</button>
    <button id="mode-b" onclick="setMode('B')">B: Spatial graph (hard cut)</button>
    <button id="mode-c" onclick="setMode('C')">C: Hybrid supported</button>
  </div>
</header>
<main>
  <div class="stage" id="stage">
    <div class="hud" id="hud"></div>
  </div>
  <div class="side">
    <div class="graph-panel" id="graph-panel"></div>
    <div class="legend">Green = strong evidence bridge (local radiance cross-dissolve, support-gated). Blue = weak bridge (direct photo crossfade, no synthesized pixels). Current node glows.</div>
    <h3>Reachable from here</h3>
    <div class="neighbor-list" id="neighbor-list"></div>
    <div class="metrics" id="metrics"></div>
  </div>
</main>
<script>
const DATA = __DATA_JSON__;
let mode = 'C';
let currentIndex = 0;
const order = DATA.nodes.map(n => n.name);

function neighborsOf(name) {
  return DATA.edges.filter(e => e.from === name || e.to === name).map(e => ({
    other: e.from === name ? e.to : e.from, edge: e,
  }));
}

function setMode(next) {
  mode = next;
  ['a','b','c'].forEach(m => document.getElementById('mode-'+m).classList.toggle('active', m === next.toLowerCase()));
  render();
}

function stageImage(src, opacity) {
  const img = document.createElement('img');
  img.className = 'layer';
  img.src = src;
  img.style.opacity = opacity;
  return img;
}

function clearStage() {
  const stage = document.getElementById('stage');
  [...stage.querySelectorAll('img.layer')].forEach(el => el.remove());
}

function showAnchor(name) {
  clearStage();
  const stage = document.getElementById('stage');
  stage.appendChild(stageImage('photos/' + name, 1));
  document.getElementById('hud').innerHTML = '<b>CAPTURED</b> photograph &middot; ' + name + '<br>You are standing where this photo was taken.';
}

async function playBridge(edge, reverse) {
  const frames = reverse ? [...edge.bridgeFrames].reverse() : edge.bridgeFrames;
  const stage = document.getElementById('stage');
  clearStage();
  for (const frame of frames) {
    clearStage();
    stage.appendChild(stageImage(frame.file, 1));
    document.getElementById('hud').innerHTML =
      '<b>RECONSTRUCTED</b> (support-gated)<br>mean support ' + frame.meanSupportScore.toFixed(2) +
      ' &middot; supported ' + frame.supportedPercent.toFixed(0) + '% / weak ' + frame.weaklySupportedPercent.toFixed(0) +
      '% / unsupported ' + frame.unsupportedPercent.toFixed(0) + '%';
    await new Promise(r => setTimeout(r, 220));
  }
}

async function playCrossfade(fromName, toName) {
  const stage = document.getElementById('stage');
  clearStage();
  const a = stageImage('photos/' + fromName, 1);
  const b = stageImage('photos/' + toName, 0);
  stage.appendChild(a); stage.appendChild(b);
  document.getElementById('hud').innerHTML = '<b>WEAK BRIDGE</b> &middot; handing off to next captured photo (no synthesized pixels).';
  for (let step = 0; step <= 10; step++) {
    a.style.opacity = 1 - step / 10;
    b.style.opacity = step / 10;
    await new Promise(r => setTimeout(r, 60));
  }
}

async function moveTo(targetName, viaEdge) {
  if (mode === 'C' && viaEdge) {
    if (viaEdge.classification === 'strong' && viaEdge.bridgeFrames.length) {
      const reverse = viaEdge.to !== targetName;
      await playBridge(viaEdge, reverse);
    } else {
      await playCrossfade(order[currentIndex], targetName);
    }
  } else if (mode === 'B' && viaEdge) {
    // Hard cut -- spatial relation exists, but no bridging experience.
  }
  currentIndex = order.indexOf(targetName);
  showAnchor(targetName);
  renderNeighbors();
  renderGraph();
}

function renderNeighbors() {
  const name = order[currentIndex];
  const list = document.getElementById('neighbor-list');
  list.innerHTML = '';
  if (mode === 'A') {
    const prevIdx = (currentIndex - 1 + order.length) % order.length;
    const nextIdx = (currentIndex + 1) % order.length;
    const prevBtn = document.createElement('button');
    prevBtn.textContent = '\u2190 Previous photo';
    prevBtn.onclick = () => moveTo(order[prevIdx], null);
    const nextBtn = document.createElement('button');
    nextBtn.textContent = 'Next photo \u2192';
    nextBtn.onclick = () => moveTo(order[nextIdx], null);
    list.appendChild(prevBtn); list.appendChild(nextBtn);
    return;
  }
  const neighbors = neighborsOf(name);
  if (!neighbors.length) {
    list.innerHTML = '<i>No spatially connected viewpoints from here.</i>';
    return;
  }
  neighbors.sort((a, b) => b.edge.meanPathSupport - a.edge.meanPathSupport);
  for (const { other, edge } of neighbors) {
    const btn = document.createElement('button');
    btn.innerHTML = other + ' <span class="tag ' + edge.classification + '">' + edge.classification + '</span>';
    btn.onclick = () => moveTo(other, edge);
    list.appendChild(btn);
  }
}

function renderGraph() {
  const panel = document.getElementById('graph-panel');
  panel.innerHTML = '';
  const rect = { w: panel.clientWidth || 300, h: panel.clientHeight || 300 };
  const pos = {};
  for (const n of DATA.nodes) pos[n.name] = { x: n.x * rect.w, y: n.y * rect.h };
  for (const e of DATA.edges) {
    const p1 = pos[e.from], p2 = pos[e.to];
    const dx = p2.x - p1.x, dy = p2.y - p1.y;
    const length = Math.hypot(dx, dy);
    const angle = Math.atan2(dy, dx) * 180 / Math.PI;
    const line = document.createElement('div');
    line.className = 'edge-line ' + e.classification;
    line.style.left = p1.x + 'px'; line.style.top = p1.y + 'px';
    line.style.width = length + 'px';
    line.style.transform = 'rotate(' + angle + 'deg)';
    panel.appendChild(line);
  }
  for (const n of DATA.nodes) {
    const dot = document.createElement('div');
    dot.className = 'node-dot' + (n.name === order[currentIndex] ? ' current' : '');
    dot.style.left = pos[n.name].x + 'px'; dot.style.top = pos[n.name].y + 'px';
    dot.title = n.name;
    dot.onclick = () => {
      const edge = DATA.edges.find(e => (e.from === order[currentIndex] && e.to === n.name) || (e.to === order[currentIndex] && e.from === n.name));
      moveTo(n.name, edge || null);
    };
    panel.appendChild(dot);
  }
}

function renderMetrics() {
  const strong = DATA.edges.filter(e => e.classification === 'strong').length;
  const weak = DATA.edges.filter(e => e.classification === 'weak').length;
  document.getElementById('metrics').textContent =
    'Nodes: ' + DATA.nodes.length + '\\nEdges: ' + DATA.edges.length +
    ' (strong ' + strong + ', weak ' + weak + ')';
}

function render() {
  showAnchor(order[currentIndex]);
  renderNeighbors();
  renderGraph();
  renderMetrics();
}

setMode('C');
</script>
"""

if __name__ == "__main__":
    main()
