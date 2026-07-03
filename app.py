"""
Roblox UGC Mesh Validator — v2
--------------------------------
Validates AI-generated meshes against Roblox's rigid-accessory spec,
with checks that match what Roblox ACTUALLY enforces:

HARD FAILS (block the item):
  - Triangle count over limit (default 4000)
  - More than one mesh OBJECT in the file (Roblox: "single mesh")
    NOTE: disconnected shells inside one mesh object are allowed.

WARNINGS (reported, do not fail by default):
  - Not watertight AFTER welding UV-seam vertices and repair attempts
    (raw GLBs almost always look "open" because vertices are split
    along texture seams; welding by position fixes false alarms)
  - Number of disconnected shells (informational)

The validator also attempts light auto-repair before judging:
merge vertices by position, drop degenerate faces, fill small holes.

POST /validate
{
  "model_url": "https://.../model.glb",
  "max_triangles": 4000,
  "fail_on_watertight": false,   # set true to make watertight a hard fail
  "debris_face_ratio": 0.005     # shells under 0.5% of faces = debris (info)
}

Response:
{
  "passed": true/false,
  "reasons": [...],      # hard failures only
  "warnings": [...],     # quality notes that don't block
  "stats": {...}
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
    return jsonify({"status": "ok", "service": "roblox-mesh-validator", "version": 2})


@app.route("/validate", methods=["POST"])
def validate():
    data = request.get_json(force=True, silent=True) or {}

    model_url = data.get("model_url")
    if not model_url:
        return jsonify({"passed": False, "reasons": ["No model_url provided"], "warnings": []}), 400

    max_triangles = int(data.get("max_triangles", DEFAULT_MAX_TRIANGLES))
    fail_on_watertight = bool(data.get("fail_on_watertight", False))
    debris_face_ratio = float(data.get("debris_face_ratio", 0.005))

    tmp_path = None
    try:
        suffix = _guess_suffix(model_url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            resp = requests.get(model_url, timeout=90, stream=True)
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=8192):
                tmp.write(chunk)

        loaded = trimesh.load(tmp_path)

        # --- Check 1: mesh OBJECT count (what Roblox's "single mesh" means) ---
        if isinstance(loaded, trimesh.Scene):
            geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            mesh_object_count = len(geoms)
            if not geoms:
                raise ValueError("No mesh geometry found in file")
            mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
        else:
            mesh_object_count = 1
            mesh = loaded

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Loaded object is not a mesh: {type(mesh)}")

        raw_tri_count = int(len(mesh.faces))
        raw_vert_count = int(len(mesh.vertices))

        # --- Repair pass (work on a copy; judge the repairable state) ---
        repaired = mesh.copy()
        # Weld vertices by position: GLB splits verts on UV seams, which
        # makes closed surfaces read as "not watertight". merge_tex/merge_norm
        # ignore UV and normal differences when welding.
        try:
            repaired.merge_vertices(merge_tex=True, merge_norm=True)
        except TypeError:
            # older trimesh signature fallback
            repaired.merge_vertices()
        repaired.update_faces(repaired.nondegenerate_faces())
        repaired.remove_unreferenced_vertices()
        try:
            trimesh.repair.fill_holes(repaired)
        except Exception:
            pass
        try:
            trimesh.repair.fix_normals(repaired)
        except Exception:
            pass

        watertight_after_repair = bool(repaired.is_watertight)

        # --- Shell analysis (informational) ---
        try:
            shells = repaired.split(only_watertight=False)
            shell_count = len(shells)
            total_faces = max(1, sum(len(s.faces) for s in shells))
            debris_count = sum(
                1 for s in shells if len(s.faces) / total_faces < debris_face_ratio
            )
        except Exception:
            shell_count = -1
            debris_count = -1

        # --- Judge ---
        reasons = []
        warnings = []

        if raw_tri_count > max_triangles:
            reasons.append(
                f"Triangle count {raw_tri_count} exceeds limit of {max_triangles}"
            )

        if mesh_object_count > 1:
            reasons.append(
                f"File contains {mesh_object_count} separate mesh objects; "
                f"Roblox requires a single mesh object"
            )

        if not watertight_after_repair:
            msg = (
                "Mesh is not fully watertight even after vertex welding and "
                "hole filling (may still pass Roblox import; check in Studio)"
            )
            if fail_on_watertight:
                reasons.append(msg)
            else:
                warnings.append(msg)

        if shell_count > 1:
            warnings.append(
                f"Mesh has {shell_count} disconnected shells "
                f"({debris_count} tiny debris shells). Allowed by Roblox within "
                f"one mesh object, but floating debris may look odd; "
                f"inspect visually in Studio."
            )

        passed = len(reasons) == 0

        return jsonify(
            {
                "passed": passed,
                "reasons": reasons,
                "warnings": warnings,
                "stats": {
                    "triangle_count": raw_tri_count,
                    "vertex_count": raw_vert_count,
                    "mesh_object_count": mesh_object_count,
                    "watertight_after_repair": watertight_after_repair,
                    "shell_count": shell_count,
                    "debris_shell_count": debris_count,
                },
            }
        )

    except requests.RequestException as e:
        # Download problems are SERVICE errors, not mesh failures
        return jsonify(
            {
                "passed": False,
                "service_error": True,
                "reasons": [f"Failed to download model: {e}"],
                "warnings": [],
            }
        ), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify(
            {
                "passed": False,
                "service_error": True,
                "reasons": [f"Validation error: {e}"],
                "warnings": [],
            }
        ), 200
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _guess_suffix(url: str) -> str:
    lower = url.lower().split("?")[0]
    for ext in (".glb", ".gltf", ".obj", ".stl", ".fbx"):
        if lower.endswith(ext):
            return ext
    return ".glb"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
