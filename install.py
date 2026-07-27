"""Vendor installer for ComfyUI-Pixal3D (envless fork).

Installs EVERY dependency into <pack>/vendor/ so the node pack is fully
self-contained: no comfy-env, no pixi, no writes to the host Python env.
On cloud providers (RunComfy, RunPod, ...) the custom_nodes folder lives on
the persisted volume, so a populated vendor/ survives pod restarts — this is
the whole point of the design.

Two dependency tiers:
  1. Pure-Python / PyPI packages  -> pinned in vendor_requirements.txt,
     installed with `pip --target vendor --no-deps` (transitives are spelled
     out explicitly in that file, so nothing can drag torch/numpy in).
  2. Compiled CUDA extensions     -> resolved from the cuda-wheels index
     (https://pozzettiandrea.github.io/cuda-wheels/v2/) by scraping each
     package's PEP 503 page and picking the wheel whose local version tag
     matches the HOST torch + CUDA + Python ABI exactly.

NEVER installed: torch, torchvision, numpy, triton — those must come from the
host ComfyUI env (the compiled wheels link against the host torch ABI).

Usage (run with ComfyUI's own python, on the machine that runs ComfyUI):
    python install.py                 # full install (auto-detect ABI)
    python install.py --check         # verify vendor imports, install nothing
    python install.py --pure-only     # skip compiled CUDA wheels
    python install.py --cuda-only     # only compiled CUDA wheels
    python install.py --dry-run       # print resolved wheels, install nothing

Env overrides:
    PIXAL3D_WHEEL_INDEX     alternate wheel index base URL
    PIXAL3D_SKIP_INSTALL=1  turn this script into a no-op (for ComfyUI-Manager)
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PACK_DIR / "vendor"
MANIFEST = VENDOR_DIR / ".pixal3d_vendor.json"
REQUIREMENTS = PACK_DIR / "vendor_requirements.txt"

WHEEL_INDEX = os.environ.get(
    "PIXAL3D_WHEEL_INDEX", "https://pozzettiandrea.github.io/cuda-wheels/v2"
).rstrip("/")

# PEP 503 name -> import name used for verification.
CUDA_PACKAGES = {
    "flash-attn": "flash_attn",
    "natten": "natten",
    "flex-gemm-ap": "flex_gemm_ap",
    "cumesh-vb": "cumesh_vb",
    "o-voxel-vb-ap": "o_voxel_vb_ap",
    "drtk": "drtk",
}

# Import names checked by --check (and after a full install).
VERIFY_IMPORTS = [
    # compiled CUDA stack
    "flash_attn", "natten", "flex_gemm_ap", "cumesh_vb", "o_voxel_vb_ap", "drtk",
    # ML stack
    "transformers", "diffusers", "accelerate", "kornia", "timm", "safetensors",
    "huggingface_hub", "tokenizers",
    # camera estimation
    "moge", "utils3d", "scipy",
    # imaging / mesh
    "cv2", "PIL", "imageio", "trimesh", "plyfile",
    # misc
    "easydict", "einops", "zstandard", "psutil",
]


def log(msg: str):
    print(f"[pixal3d-install] {msg}", flush=True)


def die(msg: str, code: int = 1):
    log(f"ERROR: {msg}")
    sys.exit(code)


# ============================================================================
# Host ABI probe
# ============================================================================

def probe_abi():
    """Return dict with python/torch/cuda/platform tags of the HOST env."""
    try:
        import torch
    except ImportError:
        die(
            "torch is not importable in this Python env. Run install.py with "
            "the SAME python that runs ComfyUI (e.g. ComfyUI's venv python)."
        )
        raise  # unreachable, keeps linters happy
    if torch.version.cuda is None:
        die(
            "Host torch has no CUDA support (CPU/MPS build). Pixal3D's "
            "compiled stack needs a CUDA torch. On a dev machine without "
            "CUDA, use --pure-only."
        )
    t_major, t_minor = torch.__version__.split(".")[:2]
    cu = torch.version.cuda.replace(".", "")
    return {
        "python": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "torch": f"torch{t_major}.{t_minor}",
        "torch_full": torch.__version__.split("+")[0],
        "cuda": f"cu{cu}",
        "platform": "win" if os.name == "nt" else "linux",
    }


# ============================================================================
# Wheel index resolution
# ============================================================================

_WHEEL_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_.]+)-(?P<version>[0-9][^-]*?)"
    r"(?:\+(?P<local>[^-]+))?"
    r"-(?P<py>[a-z0-9_.]+)-(?P<abi>[a-z0-9_.]+)-(?P<plat>[a-z0-9_.]+)\.whl$"
)


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "pixal3d-install"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def _version_key(v: str):
    try:
        from packaging.version import Version
        return Version(v)
    except Exception:
        # naive fallback: numeric-aware tuple
        return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.\-+]", v))


def resolve_wheel(pkg: str, abi: dict):
    """Scrape {index}/{pkg}/ and return the best-matching wheel URL, or None."""
    page_url = f"{WHEEL_INDEX}/{pkg}/"
    try:
        html = _fetch(page_url)
    except Exception as e:
        log(f"  {pkg}: could not fetch index page {page_url} ({e})")
        return None

    candidates = []
    for href in re.findall(r'href="([^"]+)"', html):
        clean = href.split("#")[0]
        fname = clean.rsplit("/", 1)[-1]
        # URL-decode the common case (+ encoded as %2B)
        fname = fname.replace("%2B", "+")
        m = _WHEEL_RE.match(fname)
        if not m:
            continue
        d = m.groupdict()

        # platform filter
        if abi["platform"] == "linux" and "manylinux" not in d["plat"]:
            continue
        if abi["platform"] == "win" and "win_amd64" not in d["plat"]:
            continue
        if "x86_64" not in d["plat"] and "win_amd64" not in d["plat"]:
            continue

        # python tag filter (accept exact cpXY, abi3, or pure-python py3)
        if d["py"] not in (abi["python"], "py3") and d["abi"] != "abi3":
            continue

        # ABI local-tag filter: cu + torch must match exactly when present.
        local = (d["local"] or "").lower()
        cu_m = re.search(r"cu(\d+)", local)
        torch_m = re.search(r"torch(\d+\.\d+)", local)
        if cu_m and f"cu{cu_m.group(1)}" != abi["cuda"]:
            continue
        if torch_m and f"torch{torch_m.group(1)}" != abi["torch"]:
            continue
        # Wheels with NO local tag are only trustworthy if pure-python.
        if not local and d["py"] != "py3" and d["abi"] != "abi3":
            continue

        url = clean if clean.startswith("http") else page_url + clean
        candidates.append((d["version"], bool(local), url, fname))

    if not candidates:
        return None
    # Prefer ABI-tagged wheels, then highest version.
    candidates.sort(key=lambda c: (c[1], _version_key(c[0])))
    return candidates[-1][2]


# ============================================================================
# pip helpers
# ============================================================================

def pip_install(args: list, dry_run: bool = False) -> bool:
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--target", str(VENDOR_DIR),
        "--no-deps", "--upgrade", "--no-warn-script-location",
        *args,
    ]
    log("  $ pip install --target vendor --no-deps " + " ".join(args))
    if dry_run:
        return True
    r = subprocess.run(cmd)
    return r.returncode == 0


def install_pure(dry_run: bool) -> bool:
    log(f"Installing pure-Python deps from {REQUIREMENTS.name} -> vendor/")
    return pip_install(["-r", str(REQUIREMENTS)], dry_run=dry_run)


def install_cuda(abi: dict, dry_run: bool) -> bool:
    log(
        f"Resolving compiled wheels for {abi['python']} / {abi['torch']} / "
        f"{abi['cuda']} / {abi['platform']} from {WHEEL_INDEX}"
    )
    ok = True
    for pkg in CUDA_PACKAGES:
        url = resolve_wheel(pkg, abi)
        if url is None:
            log(
                f"  {pkg}: NO matching wheel for your ABI. Check "
                f"{WHEEL_INDEX}/{pkg}/ for available torch/CUDA combos — you "
                f"may need a different machine image or a source build."
            )
            ok = False
            continue
        log(f"  {pkg}: {url.rsplit('/', 1)[-1]}")
        if not pip_install([url], dry_run=dry_run):
            log(f"  {pkg}: pip install FAILED")
            ok = False
    return ok


# ============================================================================
# Verification
# ============================================================================

def verify() -> bool:
    """Import every vendored top-level module in a clean subprocess."""
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(VENDOR_DIR)!r})\n"
        "import torch  # host torch first, compiled exts link against it\n"
        "results = {}\n"
        f"for mod in {VERIFY_IMPORTS!r}:\n"
        "    try:\n"
        "        __import__(mod)\n"
        "        results[mod] = 'ok'\n"
        "    except Exception as e:\n"
        "        results[mod] = f'{type(e).__name__}: {e}'\n"
        "print(json.dumps(results))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        log(f"verify subprocess crashed:\n{r.stderr[-2000:]}")
        return False
    try:
        results = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        log(f"verify produced unparseable output:\n{r.stdout[-2000:]}")
        return False
    ok = True
    for mod, status in results.items():
        if status == "ok":
            log(f"  OK      {mod}")
        else:
            log(f"  MISSING {mod}: {status}")
            ok = False
    return ok


def write_manifest(abi: dict):
    MANIFEST.write_text(json.dumps({
        "python": abi["python"],
        "torch": abi["torch_full"],
        "cuda": abi["cuda"],
        "platform": abi["platform"],
        "index": WHEEL_INDEX,
    }, indent=2))
    log(f"wrote {MANIFEST.relative_to(PACK_DIR)}")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="ComfyUI-Pixal3D vendor installer")
    ap.add_argument("--check", action="store_true", help="verify vendor imports only")
    ap.add_argument("--pure-only", action="store_true")
    ap.add_argument("--cuda-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if os.environ.get("PIXAL3D_SKIP_INSTALL") == "1":
        log("PIXAL3D_SKIP_INSTALL=1 -> skipping")
        return

    if args.check:
        sys.exit(0 if verify() else 1)

    VENDOR_DIR.mkdir(exist_ok=True)

    if platform.system() == "Darwin":
        log(
            "macOS detected: no CUDA stack here. Installing pure-Python deps "
            "only so you can lint/edit; run the full install on the GPU pod."
        )
        install_pure(args.dry_run)
        return

    abi = probe_abi()
    log(f"host ABI: {abi}")

    ok = True
    if not args.cuda_only:
        ok = install_pure(args.dry_run) and ok
    if not args.pure_only:
        ok = install_cuda(abi, args.dry_run) and ok

    if args.dry_run:
        log("dry run complete.")
        return

    if ok:
        write_manifest(abi)
        log("verifying imports...")
        ok = verify()

    if ok:
        log("vendor install COMPLETE. Restart ComfyUI.")
    else:
        die("vendor install finished WITH ERRORS — see MISSING lines above.")


if __name__ == "__main__":
    main()
