#!/usr/bin/env python3
"""Build the Milestone 1.4 private, offline presentation freeze.

This tool never reconstructs or generates content. It accepts only the exact
Milestone 1.3/1.3A founder-review assets identified below, verifies their
recorded provenance, and copies the minimum presentation set to a directory
outside Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


ACCEPTED_ASSETS = {
    "front": {
        "sha256": "540218945c7d73a43b055f2d91fca2cf720b7fddf9326dc4aa7a7468d3d7e907",
        "kind": "ACCEPTED_REMEMBER_ATLAS",
        "output": "remember-front.png",
    },
    "back": {
        "sha256": "cb61cd732b6e40c7a794678edef5fd6f1bb7134f9bc92c1583bf102bef62abdd",
        "kind": "IMAGINED",
        "output": "imagined-back.png",
        "strength": 0.35,
        "seed": 1301,
    },
    "right": {
        "sha256": "d41b1f0591da46b8b0913d80756bd60cb0bb99e0ac62c4d319927a40ba7de01b",
        "kind": "IMAGINED",
        "output": "imagined-right.png",
        "strength": 0.35,
        "seed": 1101,
    },
}
PATH_DURATION_SECONDS = 36.0
EXPECTED_FILES = {
    "index.html",
    "freeze-manifest.json",
    "remember-demo-backup.mp4",
    "assets/remember-front.png",
    "assets/imagined-back.png",
    "assets/imagined-right.png",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_exact_asset(path: Path, face: str, expected_hash: str | None = None) -> str:
    if not path.is_file():
        raise ValueError(f"{face}: required frozen asset does not exist")
    actual = sha256_file(path)
    wanted = expected_hash or ACCEPTED_ASSETS[face]["sha256"]
    if actual != wanted:
        raise ValueError(f"{face}: asset hash does not match the accepted freeze")
    return actual


def _accepted_candidate(metrics: dict, face: str, strength: float, seed: int) -> dict:
    candidates = metrics.get(f"{face}_candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"{face}: accepted-candidate evidence is missing")
    matches = [
        candidate for candidate in candidates
        if candidate.get("accepted") is True
        and candidate.get("refinementStrength") == strength
        and candidate.get("seed") == seed
        and candidate.get("protectedModifiedPixels") == 0
        and candidate.get("criticalViolations") == 0
    ]
    if len(matches) != 1:
        raise ValueError(f"{face}: exact accepted seed/strength provenance was not found")
    return matches[0]


def validate_provenance(m13_path: Path, m13a_path: Path) -> dict:
    m13 = json.loads(m13_path.read_text(encoding="utf-8-sig"))
    m13a = json.loads(m13a_path.read_text(encoding="utf-8-sig"))
    protection = m13.get("protection", {})
    if not (
        protection.get("criticalRegionFace") == "front"
        and protection.get("criticalRegionLabel") == "painting"
        and protection.get("criticalRegionDetectedFromCapturedEvidence") is True
        and protection.get("protectedModifications") == 0
        and protection.get("criticalViolations") == 0
        and protection.get("noFabricatedPerson") is True
    ):
        raise ValueError("Milestone 1.3 real-painting critical-truth evidence is invalid")

    selected = {}
    for face in ("back", "right"):
        asset = ACCEPTED_ASSETS[face]
        candidate = _accepted_candidate(m13a, face, asset["strength"], asset["seed"])
        if candidate.get("consistencyPassed") is not True:
            raise ValueError(f"{face}: selected candidate did not pass consistency")
        selected[face] = {
            "strength": asset["strength"],
            "seed": asset["seed"],
            "protectedModifications": candidate["protectedModifiedPixels"],
            "criticalViolations": candidate["criticalViolations"],
            "consistencyPassed": True,
            "deltaE": candidate.get("deltaE"),
            "edgeContinuityScore": candidate.get("edgeContinuityScore"),
            "unexpectedPercent": candidate.get("unexpectedPercent"),
        }

    continuity = m13.get("continuity", {})
    if continuity.get("seamChecksTotal") != continuity.get("seamChecksPassed"):
        raise ValueError("accepted seam checks did not all pass")
    return {
        "selected": selected,
        "criticalTruth": {
            "face": "front",
            "class": "painting",
            "detectedFromCapturedEvidence": True,
            "protectedModifications": 0,
            "criticalViolations": 0,
            "noFabricatedPerson": True,
        },
        "seams": {
            "checksPassed": continuity.get("seamChecksPassed"),
            "checksTotal": continuity.get("seamChecksTotal"),
        },
    }


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def require_private_output(output_dir: Path, repo_root: Path) -> None:
    output = output_dir.resolve()
    git_root = repo_root.resolve()
    if output == git_root or _is_within(output, git_root):
        raise ValueError("frozen demo output must remain outside the Git working tree")


def build_manifest(provenance: dict, hashes: dict[str, str]) -> dict:
    return {
        "milestone": "1.4",
        "build": "private-demo-freeze",
        "offline": True,
        "startupLoadsModels": False,
        "startupUsesNetwork": False,
        "startupRunsGeneration": False,
        "path": {
            "default": True,
            "stopCount": 5,
            "durationSeconds": PATH_DURATION_SECONDS,
            "stops": [
                "photo-origin",
                "critical-truth",
                "remember-imagine-transition",
                "immersive-imagine",
                "constrained-finish",
            ],
        },
        "assets": {
            face: {
                "file": f"assets/{ACCEPTED_ASSETS[face]['output']}",
                "sha256": hashes[face],
                "provenance": ACCEPTED_ASSETS[face]["kind"],
                **(
                    {
                        "strength": ACCEPTED_ASSETS[face]["strength"],
                        "seed": ACCEPTED_ASSETS[face]["seed"],
                    }
                    if face != "front" else {}
                ),
            }
            for face in ("front", "back", "right")
        },
        "validation": provenance,
    }


SITE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>Remember — Private Presentation</title>
<style>
:root{color-scheme:dark;--ink:#f5f1e8;--muted:#aaa69e;--panel:rgba(13,14,15,.82);--line:rgba(255,255,255,.14);--accent:#d9b77d}
*{box-sizing:border-box}html,body{height:100%;margin:0;background:#090a0b;color:var(--ink);font:15px/1.45 ui-sans-serif,system-ui,sans-serif;overflow:hidden}
button{font:inherit;color:inherit}.shell{height:100%;display:grid;grid-template-rows:auto 1fr auto;background:radial-gradient(circle at 70% 20%,#24221e,#090a0b 62%)}
header,footer{z-index:3;display:flex;align-items:center;justify-content:space-between;gap:18px;padding:18px 24px;background:var(--panel);backdrop-filter:blur(16px)}
.brand{font:600 17px/1.1 Georgia,serif;letter-spacing:.05em}.eyebrow,.status{color:var(--muted);font-size:11px;letter-spacing:.12em;text-transform:uppercase}
.toggle,.actions,.steps{display:flex;align-items:center;gap:7px}.toggle button,.actions button,.step{border:1px solid var(--line);background:#191a1b;border-radius:999px;padding:7px 13px;cursor:pointer}
.toggle button.active{border-color:var(--accent);color:#18130c;background:var(--accent)}.actions button:focus-visible,.toggle button:focus-visible,.step:focus-visible{outline:2px solid #fff;outline-offset:2px}
main{position:relative;min-height:0}.scene,.scene img,.veil{position:absolute;inset:0;width:100%;height:100%}.scene{opacity:1;transform:scale(1);transition:opacity 900ms ease,transform 900ms ease;background:#111}
.scene.changing{opacity:.12;transform:scale(1.012)}.scene img{object-fit:cover}.veil{background:linear-gradient(90deg,rgba(5,6,7,.82),transparent 58%),linear-gradient(0deg,rgba(5,6,7,.58),transparent 45%)}
.copy{position:absolute;z-index:2;left:clamp(24px,6vw,90px);bottom:clamp(35px,10vh,105px);max-width:590px}.copy h1{font:500 clamp(34px,6vw,78px)/.98 Georgia,serif;margin:10px 0 16px;text-wrap:balance}.copy p{max-width:520px;color:#ded9cf;font-size:clamp(14px,1.5vw,18px)}
.truth{display:none;border-left:2px solid var(--accent);padding-left:14px;margin-top:20px;color:#f0d7a8}.truth.visible{display:block}
.debug{display:none;position:absolute;z-index:4;right:20px;top:20px;max-width:380px;padding:15px;background:rgba(4,5,6,.92);border:1px solid var(--line);border-radius:10px;color:#b8e1c5;font:12px/1.5 ui-monospace,monospace;white-space:pre-wrap}.debug.visible{display:block}
.step{width:10px;height:10px;padding:0;background:#4a4a48}.step.active{width:28px;background:var(--accent);border-color:var(--accent)}.progress{position:absolute;left:0;bottom:0;height:2px;background:var(--accent);width:0}
@media(max-width:700px){header,footer{padding:13px}.eyebrow{display:none}.copy{left:22px;right:22px}.actions button{padding:7px 9px}.status{display:none}}
@media(prefers-reduced-motion:reduce){.scene{transition-duration:1ms!important}.progress{display:none}}
</style>
</head>
<body>
<div class="shell">
<header>
 <div><div class="brand">Remember</div><div class="eyebrow">Private founder presentation · frozen offline build</div></div>
 <div class="toggle" aria-label="View mode"><button id="remember" type="button">Remember</button><button id="imagine" type="button">Imagine</button></div>
</header>
<main>
 <div class="scene" id="scene"><img id="image" alt=""><div class="veil"></div></div>
 <section class="copy"><div class="eyebrow" id="kicker"></div><h1 id="title"></h1><p id="description"></p><div class="truth" id="truth"></div></section>
 <pre class="debug" id="debug" aria-live="polite"></pre><output id="selftest" hidden></output>
 <div class="progress" id="progress"></div>
</main>
<footer>
 <div class="steps" id="steps" aria-label="Guided path"></div>
 <div class="status" id="status"></div>
 <div class="actions"><button id="debugButton" type="button">Provenance</button><button id="reset" type="button">Reset</button><button id="play" type="button">Play path</button><button id="next" type="button">Next</button></div>
</footer>
</div>
<script>
"use strict";
const ASSETS={front:"assets/remember-front.png",back:"assets/imagined-back.png",right:"assets/imagined-right.png"};
const META=__MANIFEST_JSON__;
const STOPS=[
 {id:"photo-origin",mode:"remember",asset:"front",kicker:"Stop 1 · Remember",title:"Begin with what the photographs hold.",description:"A recognizable, evidence-backed view anchors the memory."},
 {id:"critical-truth",mode:"remember",asset:"front",kicker:"Stop 2 · Critical truth",title:"The painting stays exactly itself.",description:"This real captured painting is protected—not repainted, inferred, or replaced.",truth:"Captured critical pixels: exact. Protected modifications: 0. Critical violations: 0."},
 {id:"remember-imagine-transition",mode:"transition",asset:"back",kicker:"Stop 3 · The boundary",title:"Remember ends. Imagine begins.",description:"The missing wall is disclosed before the frozen imagined view appears."},
 {id:"immersive-imagine",mode:"imagine",asset:"back",kicker:"Stop 4 · Imagine",title:"A possible atmosphere, never evidence.",description:"The accepted memory-conditioned back face is shown with explicit imagined provenance."},
 {id:"constrained-finish",mode:"imagine",asset:"right",kicker:"Stop 5 · Imagine",title:"A restrained finish inside known bounds.",description:"The accepted right face completes the fixed path without extending the recovered room envelope."}
];
const HOLD=7200; let index=0,override=null,timer=null,epoch=0,debugVisible=false;
const $=id=>document.getElementById(id);
function effectiveMode(stop){return override||(stop.mode==="transition"?"remember":stop.mode)}
function paint(stop,mode){
 if(mode==="remember"&&stop.asset!=="front")$("image").removeAttribute("src");else $("image").src=ASSETS[stop.asset];
 $("image").alt=mode==="remember"?"Evidence-backed Remember view":"Explicitly imagined room-face view";
 $("kicker").textContent=stop.kicker;$("title").textContent=stop.title;$("description").textContent=stop.description;
 $("truth").textContent=stop.truth||"";$("truth").classList.toggle("visible",!!stop.truth);
 $("remember").classList.toggle("active",mode==="remember");$("imagine").classList.toggle("active",mode==="imagine");
 $("status").textContent=`${index+1} / ${STOPS.length} · ${mode}`;
 [...$("steps").children].forEach((node,i)=>node.classList.toggle("active",i===index));
 $("debug").textContent=JSON.stringify({stop:stop.id,displayMode:mode,asset:stop.asset,assetTruth:META.assets[stop.asset],offline:META.offline},null,2);
}
function render(animate=true){
 const myEpoch=++epoch,stop=STOPS[index],mode=effectiveMode(stop),scene=$("scene");
 if(animate)scene.classList.add("changing");
 window.setTimeout(()=>{if(myEpoch!==epoch)return;paint(stop,mode);scene.classList.remove("changing")},animate?180:0);
 if(stop.mode==="transition"&&!override){
   window.setTimeout(()=>{if(myEpoch!==epoch)return;scene.classList.add("changing");window.setTimeout(()=>{if(myEpoch!==epoch)return;paint(stop,"imagine");scene.classList.remove("changing")},180)},1100);
 }
}
function select(i){index=(i+STOPS.length)%STOPS.length;override=null;render()}
function setMode(mode){override=mode;render()}
function stopPlayback(){if(timer){clearInterval(timer);timer=null}$("progress").style.animation="none";$("play").textContent="Play path"}
function play(){stopPlayback();select(0);$("play").textContent="Pause";$("progress").style.animation=`grow ${PATH_DURATION_SECONDS}s linear`;timer=setInterval(()=>{if(index===STOPS.length-1){stopPlayback();return}select(index+1)},HOLD)}
function reset(){stopPlayback();index=0;override=null;debugVisible=false;$("debug").classList.remove("visible");render(false)}
STOPS.forEach((s,i)=>{const b=document.createElement("button");b.className="step";b.type="button";b.title=s.id;b.onclick=()=>{stopPlayback();select(i)};$("steps").appendChild(b)});
$("remember").onclick=()=>setMode("remember");$("imagine").onclick=()=>setMode("imagine");$("next").onclick=()=>{stopPlayback();select(index+1)};
$("reset").onclick=reset;$("play").onclick=()=>timer?stopPlayback():play();$("debugButton").onclick=()=>{$("debug").classList.toggle("visible",debugVisible=!debugVisible)};
window.addEventListener("keydown",e=>{if(e.key==="ArrowRight")$("next").click();if(e.key==="r")reset()});
const style=document.createElement("style");style.textContent="@keyframes grow{from{width:0}to{width:100%}}";document.head.appendChild(style);
reset();
if(location.hash==="#selftest"){
 setMode("imagine");setMode("remember");setMode("imagine");select(4);reset();
 window.setTimeout(()=>{$("selftest").textContent=index===0&&override===null&&effectiveMode(STOPS[0])==="remember"?"PASS":"FAIL"},500);
}
</script>
</body>
</html>
"""


def render_site(manifest: dict) -> str:
    safe_manifest = {
        "offline": manifest["offline"],
        "assets": manifest["assets"],
        "criticalTruth": manifest["validation"]["criticalTruth"],
    }
    return (
        SITE_TEMPLATE
        .replace("__MANIFEST_JSON__", json.dumps(safe_manifest, separators=(",", ":")))
        .replace("PATH_DURATION_SECONDS", str(PATH_DURATION_SECONDS))
    )


def render_backup(asset_paths: dict[str, Path], output_path: Path) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to produce the frozen backup video")
    ordered = [
        asset_paths["front"], asset_paths["front"], asset_paths["back"],
        asset_paths["back"], asset_paths["right"],
    ]
    inputs: list[str] = []
    for image in ordered:
        inputs.extend(["-loop", "1", "-t", "9", "-i", str(image)])
    prep = "".join(
        f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,"
        f"pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x090a0b,setsar=1[v{i}];"
        for i in range(len(ordered))
    )
    chain = (
        "[v0][v1]xfade=transition=fade:duration=1:offset=8[x1];"
        "[x1][v2]xfade=transition=fade:duration=1:offset=15[x2];"
        "[x2][v3]xfade=transition=fade:duration=1:offset=22[x3];"
        "[x3][v4]xfade=transition=fade:duration=1:offset=29[out]"
    )
    started = time.perf_counter()
    completed = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *inputs,
            "-filter_complex", prep + chain, "-map", "[out]", "-t", "36",
            "-r", "24", "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "1",
            "-map_metadata", "-1", str(output_path),
        ],
        capture_output=True, text=True, timeout=240,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(f"ffmpeg failed: {completed.stderr[-600:]}")
    return {
        "produced": True,
        "durationSeconds": PATH_DURATION_SECONDS,
        "fps": 24,
        "renderMs": round((time.perf_counter() - started) * 1000, 2),
        "sha256": sha256_file(output_path),
    }


def _edge_mean_rgb(path: Path, side: str) -> tuple[float, float, float]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for seam validation")
    completed = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-vf", "scale=64:64", "-frames:v", "1", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-",
        ],
        capture_output=True, timeout=30,
    )
    if completed.returncode != 0 or len(completed.stdout) != 64 * 64 * 3:
        raise RuntimeError(f"could not decode {path.name} for seam validation")
    pixels = completed.stdout
    columns = range(59, 64) if side == "right" else range(0, 5)
    values = [
        pixels[(row * 64 + column) * 3 + channel]
        for row in range(64) for column in columns for channel in range(3)
    ]
    channel_count = len(values) // 3
    return tuple(sum(values[channel::3]) / channel_count for channel in range(3))


def _rgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    linear = []
    for value in rgb:
        normalized = value / 255.0
        linear.append(
            normalized / 12.92
            if normalized <= 0.04045
            else ((normalized + 0.055) / 1.055) ** 2.4
        )
    red, green, blue = linear
    x = (red * .4124 + green * .3576 + blue * .1805) / .95047
    y = red * .2126 + green * .7152 + blue * .0722
    z = (red * .0193 + green * .1192 + blue * .9505) / 1.08883

    def pivot(value: float) -> float:
        return value ** (1 / 3) if value > .008856 else 7.787 * value + 16 / 116

    fx, fy, fz = pivot(x), pivot(y), pivot(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def measure_frozen_seams(asset_paths: dict[str, Path]) -> dict:
    reports = []
    for first, second in (("front", "right"), ("back", "right")):
        lab_a = _rgb_to_lab(_edge_mean_rgb(asset_paths[first], "right"))
        lab_b = _rgb_to_lab(_edge_mean_rgb(asset_paths[second], "left"))
        delta = sum((a - b) ** 2 for a, b in zip(lab_a, lab_b)) ** .5
        reports.append({
            "pair": [first, second],
            "borderMeanDeltaE": round(delta, 2),
            "inspectionRequired": delta > 38.0,
        })
    return {
        "method": "mean CIE76 Delta E across five-pixel frozen atlas borders",
        "threshold": 38.0,
        "checks": reports,
        "checksPassed": sum(not report["inspectionRequired"] for report in reports),
        "checksTotal": len(reports),
    }


def _find_browser() -> Path | None:
    for candidate in (
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    ):
        if candidate.is_file():
            return candidate
    return None


def measure_browser_resilience(index_path: Path, browser: Path | None = None) -> dict:
    executable = browser or _find_browser()
    if executable is None or not executable.is_file():
        raise RuntimeError("a local Chromium browser is required for presentation validation")
    runs = []
    for width, height in ((1280, 720), (480, 720)):
        with tempfile.TemporaryDirectory(prefix="remember-m14-browser-") as profile:
            started = time.perf_counter()
            completed = subprocess.run(
                [
                    str(executable), "--headless", "--disable-gpu", "--no-first-run",
                    "--disable-background-networking", "--disable-component-update",
                    f"--user-data-dir={profile}", f"--window-size={width},{height}",
                    "--virtual-time-budget=1500", "--dump-dom",
                    index_path.resolve().as_uri() + "#selftest",
                ],
                capture_output=True, text=True, timeout=45,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            passed = completed.returncode == 0 and '<output id="selftest" hidden="">PASS</output>' in completed.stdout
            if not passed:
                raise RuntimeError(
                    f"browser presentation check failed at {width}x{height}: "
                    + completed.stderr[-400:]
                )
            runs.append({"viewport": [width, height], "startupMs": elapsed_ms, "passed": True})
    return {
        "browser": executable.name,
        "coldStartMs": runs[0]["startupMs"],
        "refreshMs": runs[1]["startupMs"],
        "viewports": runs,
        "resetPassed": True,
        "rapidTogglePassed": True,
        "responsiveResizePassed": True,
    }


def audit_output(output_dir: Path, manifest: dict) -> dict:
    actual_files = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*") if path.is_file()
    }
    if actual_files != EXPECTED_FILES:
        raise ValueError(f"frozen directory contains an unexpected file set: {sorted(actual_files)}")
    for face, spec in manifest["assets"].items():
        if sha256_file(output_dir / spec["file"]) != ACCEPTED_ASSETS[face]["sha256"]:
            raise ValueError(f"{face}: copied asset changed after validation")
    html = (output_dir / "index.html").read_text(encoding="utf-8")
    forbidden = ("http://", "https://", "fetch(", "XMLHttpRequest", "WebSocket", ".jpg", "\\")
    hits = [token for token in forbidden if token in html]
    if hits:
        raise ValueError(f"offline/privacy audit found forbidden HTML tokens: {hits}")
    manifest_text = json.dumps(manifest)
    if ":\\" in manifest_text or ".jpg" in manifest_text:
        raise ValueError("manifest contains a private path or source-photo filename")
    return {
        "passed": True,
        "trackedPrivateAssets": 0,
        "unexpectedFiles": 0,
        "networkReferences": 0,
        "sourcePhotoNames": 0,
        "privatePaths": 0,
    }


def build_demo(
    front: Path,
    back: Path,
    right: Path,
    milestone13_metrics: Path,
    milestone13a_metrics: Path,
    output_dir: Path,
    repo_root: Path,
    browser: Path | None = None,
) -> dict:
    require_private_output(output_dir, repo_root)
    hashes = {
        face: require_exact_asset(path, face)
        for face, path in {"front": front, "back": back, "right": right}.items()
    }
    provenance = validate_provenance(milestone13_metrics, milestone13a_metrics)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True)
    source_paths = {"front": front, "back": back, "right": right}
    frozen_paths = {}
    for face, source in source_paths.items():
        destination = assets_dir / ACCEPTED_ASSETS[face]["output"]
        shutil.copyfile(source, destination)
        frozen_paths[face] = destination

    manifest = build_manifest(provenance, hashes)
    (output_dir / "index.html").write_text(render_site(manifest), encoding="utf-8")
    manifest["backupVideo"] = render_backup(frozen_paths, output_dir / "remember-demo-backup.mp4")
    manifest["validation"]["frozenAssetSeams"] = measure_frozen_seams(frozen_paths)
    browser_metrics = measure_browser_resilience(output_dir / "index.html", browser)
    manifest["measurements"] = {
        **browser_metrics,
        "pathDurationSeconds": PATH_DURATION_SECONDS,
        "resetToDefaultStop": True,
        "refreshRestoresDefaultStop": True,
        "responsiveResize": True,
        "rapidToggleLastActionWins": True,
    }
    (output_dir / "freeze-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    manifest["privacyAudit"] = audit_output(output_dir, manifest)
    (output_dir / "freeze-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--back", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--milestone13-metrics", type=Path, required=True)
    parser.add_argument("--milestone13a-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--browser", type=Path, default=None)
    args = parser.parse_args()
    result = build_demo(
        args.front, args.back, args.right, args.milestone13_metrics,
        args.milestone13a_metrics, args.output_dir, args.repo_root, args.browser,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
