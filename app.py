"""
Roblox UGC Mesh Validator
--------------------------
Small HTTP service that downloads a generated mesh (GLB/FBX/OBJ), checks it
against Roblox's rigid-accessory technical spec, and returns pass/fail +
reasons so an n8n workflow can branch on it.

Roblox rigid accessory rules this checks:
  - Triangle count under a configurable limit (default 4000)
  - Mesh is watertight (no holes / exposed backfaces)
  - Mesh is a single connected piece (Roblox requires "single mesh")

Endpoint:
  POST /validate
  {
    "model_url": "https://.../model.glb",
    "max_triangles": 4000,
    "require_watertight": true,
    "require_single_mesh": true
  }

Response:
  {
    "passed": true/false,
    "reasons": ["..."],           // empty if passed
    "stats": {
      "triangle_count": 3421,
      "watertight": true,
      "body_count": 1,
      "vertex_count": 1780
    }
  }
"""

import os
import tempfile
import traceback

import requests
import trimesh
from flask import Flask, request, jsonify

app = Flask(__name__)

DEFAULT_MAX_TRIANGLES = 4000


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "roblox-mesh-validator"})


@app.route("/validate", methods=["POST"])
def validate():
    data = request.get_json(force=True, silent=True) or {}

    model_url = data.get("model_url")
    if not model_url:
        return jsonify({"passed": False, "reasons": ["No model_url provided"]}), 400

    max_triangles = int(data.get("max_triangles", DEFAULT_MAX_TRIANGLES))
    require_watertight = bool(data.get("require_watertight", True))
    require_single_mesh = bool(data.get("require_single_mesh", True))

    tmp_path = None
    try:
        # Download the model to a temp file (trimesh needs a file extension
        # to pick the right loader)
        suffix = _guess_suffix(model_url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            resp = requests.get(model_url, timeout=60, stream=True)
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=8192):
                tmp.write(chunk)

        mesh = _load_as_single_mesh(tmp_path)

        reasons = []
        tri_count = int(len(mesh.faces))
        vert_count = int(len(mesh.vertices))

        # Body count: how many disconnected pieces of geometry exist.
        # Roblox rigid accessories must be a single mesh.
        try:
            body_count = mesh.body_count
        except Exception:
            # Fallback via connected components on the face adjacency graph
            body_count = len(mesh.split(only_watertight=False))

        watertight = bool(mesh.is_watertight)

        if tri_count > max_triangles:
            reasons.append(
                f"Triangle count {tri_count} exceeds limit of {max_triangles}"
            )

        if require_watertight and not watertight:
            reasons.append(
                "Mesh is not watertight (has holes or exposed backfaces)"
            )

        if require_single_mesh and body_count > 1:
            reasons.append(
                f"Mesh has {body_count} disconnected pieces; must be a single mesh"
            )

        passed = len(reasons) == 0

        return jsonify(
            {
                "passed": passed,
                "reasons": reasons,
                "stats": {
                    "triangle_count": tri_count,
                    "vertex_count": vert_count,
                    "watertight": watertight,
                    "body_count": int(body_count),
                },
            }
        )

    except requests.RequestException as e:
        return jsonify(
            {"passed": False, "reasons": [f"Failed to download model: {e}"]}
        ), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify(
            {"passed": False, "reasons": [f"Validation error: {e}"]}
        ), 200
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _guess_suffix(url: str) -> str:
    lower = url.lower().split("?")[0]
    for ext in (".glb", ".gltf", ".fbx", ".obj", ".stl"):
        if lower.endswith(ext):
            return ext
    # Default to glb since that's Tripo's most common output
    return ".glb"


def _load_as_single_mesh(path: str) -> trimesh.Trimesh:
    """
    Load a file with trimesh and flatten it to a single Trimesh object.
    Scenes (multi-node GLB/FBX files) get concatenated into one mesh so
    triangle count and watertightness reflect the whole visible model.
    """
    loaded = trimesh.load(path, force="mesh")

    if isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
        if not geoms:
            raise ValueError("No geometry found in file")
        mesh = trimesh.util.concatenate(geoms)
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded object is not a mesh: {type(mesh)}")

    return mesh


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
