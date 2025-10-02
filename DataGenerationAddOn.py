# ==== MERS Lab – Synthetic Data Builder (Simplified UX) ====
# Goals
# - One simple "Quick Add" block controls *where* an asset is placed, either in World or relative to another object
# - Import assets from a folder of .blend files (dropdown)
# - Table (UIList) of added items with editable params
# - Bake Motion (from frame 1) using velocity + optional relative motion
# - Depth-style point cloud export (ONE CSV for all frames)
# - Render MP4 and reset timeline to frame 1 after export/render
# Works with Blender 3.x / 4.x

bl_info = {
    "name": "MERS Lab – Synthetic Data Builder (Simplified)",
    "author": "Jack + ChatGPT",
    "version": (0, 5, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Data Generation",
    "description": "Import assets with world/relative placement, bake motion, export point clouds, render video.",
    "category": "Object",
}

import bpy
from bpy.props import (
    EnumProperty, PointerProperty, FloatVectorProperty, CollectionProperty,
    IntProperty, StringProperty, BoolProperty, FloatProperty
)
from bpy.types import Panel, PropertyGroup, Operator, UIList
from pathlib import Path
from mathutils import Vector
import csv
import os
import math

# ---------------------------------------------------------
# CONFIG: change to your asset folder containing .blend files
# ---------------------------------------------------------
ASSET_DIR = Path(r"C:/Users/jonat/Documents/Course Work/Current Courses/UROP/SyntheticDataGeneration/CustomAssets")

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def enum_assets(self, context):
    items = []
    if ASSET_DIR.exists():
        for p in sorted(ASSET_DIR.glob("*.blend")):
            name = p.stem
            items.append((name, name, f"Import {name}"))
    if not items:
        items = [("NONE", "No .blend assets found", "Put .blend files in ASSET_DIR")]
    return items


def draw_vec3_inline(layout, data, prop, title="Vector"):
    box = layout.box()
    box.use_property_split = False
    box.use_property_decorate = False
    split = box.split(factor=0.33)
    col_l = split.column(align=True)
    col_r = split.column(align=True)
    col_l.label(text=title); row = col_r.row(align=True); row.prop(data, prop, index=0, text="X")
    col_l.label(text="");    col_r.prop(data, prop, index=1, text="Y")
    col_l.label(text="");    col_r.prop(data, prop, index=2, text="Z")


def _child_names(coll):
    try: return [c.name for c in coll.children]
    except Exception: return []

def _object_names(coll):
    try: return [o.name for o in coll.objects]
    except Exception: return []


def _ensure_camera(context, props):
    cam = context.scene.camera if props.use_active_camera else props.camera_override
    if cam and cam.type == 'CAMERA':
        return cam
    bpy.ops.object.camera_add(location=(0.0, -5.0, 2.0), rotation=(math.radians(75), 0, 0))
    context.scene.camera = bpy.context.active_object
    return context.scene.camera


def _camera_ray_for_pixel(cam_obj, i, j, W, H):
    cam = cam_obj.data
    frame = cam.view_frame(scene=bpy.context.scene)  # [bl, br, tr, tl]
    bl, br, tr, tl = frame
    u = (i + 0.5) / W
    v = (j + 0.5) / H
    p_local = (1 - u) * ((1 - v) * bl + v * tl) + u * ((1 - v) * br + v * tr)
    origin = cam_obj.matrix_world.translation
    p_world = cam_obj.matrix_world @ p_local
    direction = (p_world - origin).normalized()
    return origin, direction

# ---- Bounding boxes & relative placement ----

def _world_bbox_min_max(ob):
    mw = ob.matrix_world
    corners = [mw @ Vector(c) for c in getattr(ob, 'bound_box', [])] if getattr(ob, 'bound_box', None) else []
    if not corners:
        loc = mw.translation
        return loc.copy(), loc.copy()
    xs, ys, zs = zip(*[(c.x, c.y, c.z) for c in corners])
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def _group_bbox_min_max(root):
    if not root:
        z = Vector((0, 0, 0))
        return z, z
    mins, maxs = [], []
    stack = list(root.children)
    while stack:
        ob = stack.pop()
        if ob.type != 'EMPTY':
            mn, mx = _world_bbox_min_max(ob)
            mins.append(mn); maxs.append(mx)
        stack.extend(list(ob.children))
    if not mins:
        loc = root.matrix_world.translation.copy()
        return loc, loc
    minx = Vector((min(v.x for v in mins), min(v.y for v in mins), min(v.z for v in mins)))
    maxx = Vector((max(v.x for v in maxs), max(v.y for v in maxs), max(v.z for v in maxs)))
    return minx, maxx


def _place_on_top(child_root, ref_ob, margin=0.0, align_xy=True, extra_offset=None):
    if not (child_root and ref_ob):
        return
    ref_min, ref_max = _world_bbox_min_max(ref_ob)
    ref_top_z = ref_max.z
    grp_min, grp_max = _group_bbox_min_max(child_root)
    grp_center_xy = Vector(((grp_min.x + grp_max.x)/2, (grp_min.y + grp_max.y)/2, 0.0))
    grp_bottom = Vector((grp_center_xy.x, grp_center_xy.y, grp_min.z))
    ref_center_xy = Vector(((ref_min.x + ref_max.x)/2, (ref_min.y + ref_max.y)/2, 0.0))
    tgt_bottom = Vector((ref_center_xy.x if align_xy else grp_center_xy.x,
                         ref_center_xy.y if align_xy else grp_center_xy.y,
                         ref_top_z + margin))
    delta = tgt_bottom - grp_bottom
    child_root.location += delta
    if extra_offset is not None:
        child_root.location += Vector(extra_offset)

# ---------------------------------------------------------
# Data Model
# ---------------------------------------------------------

def _item_location_update(self, context):
    if self.created_object:
        self.created_object.location = Vector(self.location)

def _item_name_update(self, context):
    if self.created_object and self.name:
        self.created_object.name = self.name



class MyItem(PropertyGroup):
    name: StringProperty(name="Name", default="", update=_item_name_update)
    asset_name: StringProperty(name="Asset", default="")
    location: FloatVectorProperty(name="Location", subtype='TRANSLATION', size=3, default=(0,0,0), update=_item_location_update)
    velocity: FloatVectorProperty(name="Velocity (m/s)", subtype='VELOCITY', size=3, default=(0,0,0))
    rel_frame: EnumProperty(name="Relative To",
                            items=[("WORLD","World","World-frame velocity"),
                                   ("OBJECT","Object","Velocity relative to a reference object")],
                            default="WORLD")
    reference_object: PointerProperty(name="Reference Object", type=bpy.types.Object)
    created_object: PointerProperty(name="Created Object", type=bpy.types.Object)



class MyAddonProperties(PropertyGroup):
    # Quick Add (EVERYTHING lives here: location + placement mode)
    my_enum: EnumProperty(name="Objects", description="Choose asset", items=enum_assets)

    # Placement-at-import block (first block for ease-of-use)
    place_rel_mode: EnumProperty(name="Place Relative",
                                 items=[("WORLD","World","Use absolute Location"),
                                        ("OBJECT","Object","Place relative to a reference object")],
                                 default="WORLD")
    location: FloatVectorProperty(name="Location (World)", subtype='TRANSLATION', size=3, default=(0,0,0))
    place_reference: PointerProperty(name="Reference", type=bpy.types.Object, description="Reference object (e.g., table)")
    place_on_top: BoolProperty(name="Snap On Top", default=True, description="Place object on reference top surface")
    place_margin: FloatProperty(name="Top Margin", default=0.01, min=0.0)
    place_align_xy: BoolProperty(name="Center XY", default=True)
    place_offset: FloatVectorProperty(name="Extra Offset", subtype='TRANSLATION', size=3, default=(0.0,0.0,0.0))

    # Motion settings used for Bake
    velocity: FloatVectorProperty(name="Velocity (m/s)", subtype='VELOCITY', size=3, default=(0,0,0))
    rel_frame: EnumProperty(name="Motion Relative To",
                            items=[("WORLD","World",""), ("OBJECT","Object","")],
                            default="WORLD")
    reference_object: PointerProperty(name="Motion Reference", type=bpy.types.Object)

    # Table
    items: CollectionProperty(type=MyItem)
    active_index: IntProperty(default=0, min=0)

    # Recording / export settings
    bake_fps: IntProperty(name="FPS", default=24, min=1, max=240)
    bake_seconds: IntProperty(name="Seconds", default=3, min=1, max=600)
    pc_width: IntProperty(name="PC Width", default=320, min=8, max=3840)
    pc_height: IntProperty(name="PC Height", default=240, min=8, max=2160)
    output_dir: StringProperty(name="Output Dir", subtype="DIR_PATH", default="//")
    file_base: StringProperty(name="File Base", default="synth_scene")
    make_video: BoolProperty(name="Also Render Video", default=True)
    video_kind: EnumProperty(name="Video", items=[("MP4","MP4 (H.264)","")], default="MP4")
    use_active_camera: BoolProperty(name="Use Active Camera", default=True)
    camera_override: PointerProperty(name="Camera", type=bpy.types.Object)

# ---------------------------------------------------------
# Operators
# ---------------------------------------------------------

class MYADDON_OT_import_selected(Operator):
    bl_idname = "myaddon.import_selected"
    bl_label = "Add Selected Asset"
    bl_description = "Append datablock named like the dropdown and place according to the first block (World/Object)"

    def execute(self, context):
        props = context.scene.my_addon_props
        name  = props.my_enum
        if name == "NONE":
            self.report({'WARNING'}, "No assets to import. Put .blend files in ASSET_DIR.")
            return {'CANCELLED'}
        libpath = ASSET_DIR / f"{name}.blend"
        if not libpath.exists():
            self.report({'ERROR'}, f"File not found: {libpath}")
            return {'CANCELLED'}

        try:
            with bpy.data.libraries.load(str(libpath), link=False) as (data_from, data_to):
                has_col = name in (data_from.collections or [])
                has_obj = name in (data_from.objects or [])
            if not (has_col or has_obj):
                self.report({'ERROR'}, f"'{name}' not found as Collection/Object in {libpath.name}.")
                return {'CANCELLED'}
            with bpy.data.libraries.load(str(libpath), link=False) as (data_from, data_to):
                if has_col: data_to.collections = [name]
                else:       data_to.objects     = [name]

            created = None
            if has_col:
                coll = bpy.data.collections.get(name)
                if coll and coll.name not in _child_names(context.scene.collection):
                    context.scene.collection.children.link(coll)
                if coll and coll.objects:
                    ctrl = bpy.data.objects.new(f"{name}_ctrl", None)
                    ctrl.empty_display_type = 'PLAIN_AXES'
                    context.scene.collection.objects.link(ctrl)
                    # parent the imported objects to controller
                    for ob in coll.objects:
                        ob.parent = ctrl
                        ob.matrix_parent_inverse = ctrl.matrix_world.inverted()
                    created = ctrl
            else:
                ob = bpy.data.objects.get(name)
                if ob:
                    if ob.name not in _object_names(context.scene.collection):
                        context.scene.collection.objects.link(ob)
                    created = ob

            if created is None:
                self.report({'WARNING'}, f"Nothing linked from {libpath.name}")
                return {'CANCELLED'}

            # ---- Placement (ONE PASS, from the first block) ----
            if props.place_rel_mode == "WORLD" or not props.place_reference:
                created.location = Vector(props.location) + Vector(props.place_offset)
            else:
                ref = props.place_reference
                if props.place_on_top:
                    _place_on_top(created, ref_ob=ref, margin=props.place_margin, align_xy=props.place_align_xy, extra_offset=Vector(props.place_offset))
                else:
                    created.location = ref.matrix_world.translation + Vector(props.place_offset)

            bpy.context.view_layer.update()

            # Add to table
            row = props.items.add()
            row.asset_name = name
            row.created_object = created
            row.name = created.name
            row.location = created.location[:]
            row.velocity = props.velocity
            row.rel_frame = props.rel_frame
            row.reference_object = props.reference_object
            props.active_index = len(props.items) - 1

        except Exception as e:
            self.report({'ERROR'}, f"Append failed: {e}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Added '{name}'")
        return {'FINISHED'}


class MYADDON_OT_bake_motion(Operator):
    bl_idname = "myaddon.bake_motion"
    bl_label = "Bake Motion"
    bl_description = "Insert location keyframes for all items using velocity and relative frame (starts at frame 1)"

    def execute(self, context):
        scn = context.scene
        props = scn.my_addon_props
        fps = props.bake_fps
        total_frames = props.bake_seconds * fps
        start = 1
        end   = start + total_frames

        scn.frame_start = start
        scn.frame_end   = end
        scn.frame_current = start

        scn.frame_set(start)
        base = {}
        ref0 = {}
        for it in props.items:
            ob = it.created_object
            if not ob: continue
            base[it.name] = ob.location.copy()
            if it.rel_frame == "OBJECT" and it.reference_object:
                ref0[it.name] = it.reference_object.matrix_world.translation.copy()
            else:
                ref0[it.name] = Vector((0,0,0))

        # Clear previous location keys in range
        for it in props.items:
            ob = it.created_object
            if not (ob and ob.animation_data and ob.animation_data.action):
                continue
            act = ob.animation_data.action
            for fc in [fc for fc in act.fcurves if fc.data_path == "location"]:
                for kp in list(fc.keyframe_points):
                    if start - 1e-6 <= kp.co.x <= end + 1e-6:
                        fc.keyframe_points.remove(kp)

        for f in range(start, end + 1):
            scn.frame_set(f)
            t = (f - start) / fps
            for it in props.items:
                ob = it.created_object
                if not ob: continue
                v = Vector(it.velocity)
                if it.rel_frame == "WORLD" or not it.reference_object:
                    pos = base[it.name] + v * t
                else:
                    ref_now = it.reference_object.matrix_world.translation
                    pos = base[it.name] + (ref_now - ref0[it.name]) + v * t
                ob.location = pos
                ob.keyframe_insert("location", frame=f)

        self.report({'INFO'}, f"Baked {len(props.items)} items from {start} to {end} at {fps} fps.")
        return {'FINISHED'}


import csv, os, math, random, re
from mathutils import Vector

def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", name).strip().strip(".")
    return name or "object"

def _gather_mesh_objects(root_obj):
    """Return all mesh children under a controller (includes root if it's a mesh)."""
    meshes = []
    if not root_obj:
        return meshes
    if root_obj.type == 'MESH':
        meshes.append(root_obj)
    stack = list(root_obj.children)
    while stack:
        ob = stack.pop()
        if ob.type == 'MESH':
            meshes.append(ob)
        stack.extend(list(ob.children))
    return meshes

def _collect_world_triangles(mesh_objs, depsgraph):
    """
    Build a list of world-space triangles [(v0,v1,v2), ...] for all evaluated meshes.
    """
    tris = []
    for ob in mesh_objs:
        ob_eval = ob.evaluated_get(depsgraph)
        me = ob_eval.to_mesh()
        if not me:
            continue
        try:
            me.calc_loop_triangles()
            mw = ob_eval.matrix_world
            verts = [mw @ v.co for v in me.vertices]
            for lt in me.loop_triangles:
                i0, i1, i2 = lt.vertices
                tris.append((verts[i0], verts[i1], verts[i2]))
        finally:
            ob_eval.to_mesh_clear()
    return tris

def _triangle_area(a: Vector, b: Vector, c: Vector) -> float:
    return (b - a).cross(c - a).length * 0.5

def _cdf_from_triangles(tris):
    cdf, total = [], 0.0
    for (a,b,c) in tris:
        total += _triangle_area(a,b,c)
        cdf.append(total)
    return cdf, total

def _sample_on_triangle(a: Vector, b: Vector, c: Vector):
    # Uniform barycentric
    r1, r2 = random.random(), random.random()
    s = math.sqrt(r1)
    u = 1.0 - s
    v = s * (1.0 - r2)
    w = s * r2
    return a*u + b*v + c*w

def _sample_surface_points(tris, N, seed=None):
    if seed is not None:
        random.seed(seed)
    if not tris:
        return [Vector((0.0,0.0,0.0))] * N
    cdf, total = _cdf_from_triangles(tris)
    pts = []
    for _ in range(N):
        r = random.random() * total
        # binary search
        lo, hi = 0, len(cdf) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if r <= cdf[mid]:
                hi = mid
            else:
                lo = mid + 1
        a, b, c = tris[lo]
        pts.append(_sample_on_triangle(a, b, c))
    return pts

# ------------------ Drop-in operator ------------------

class MYADDON_OT_export_pointclouds(bpy.types.Operator):
    bl_idname = "myaddon.export_pointclouds"
    bl_label = "Export Point Clouds (Surface, 1 Row/Frame)"
    bl_description = (
        "Uniformly sample N surface points for each object per frame (occlusion-free), "
        "writing one CSV per object with rows = frames and columns = flattened XYZ triples. "
        "Creates a new run folder and resets to frame 1 after export."
    )

    def execute(self, context):
        scn = context.scene
        props = scn.my_addon_props

        # Frame plan from your settings (start at 1)
        fps = props.bake_fps
        total_frames = props.bake_seconds * fps
        start = 1
        end   = start + total_frames

        # Output folder: <OutputDir>/<FileBase>_object_clouds/
        base_dir = bpy.path.abspath(props.output_dir)
        os.makedirs(base_dir, exist_ok=True)
        run_dir = os.path.join(base_dir, f"{props.file_base}_object_clouds")
        os.makedirs(run_dir, exist_ok=True)

        # How many points per object per frame (fallback to 1024 if the prop doesn't exist)
        N = getattr(props, "points_per_object", 512)
        # Optional reproducibility
        seed_base = getattr(props, "random_seed", 0) or None

        # Header: frame, x1,y1,z1, ... xN,yN,zN
        header = ["frame"] + [axis for i in range(1, N+1) for axis in (f"x{i}", f"y{i}", f"z{i}")]

        deps = context.evaluated_depsgraph_get()

        files = {}
        writers = {}
        try:
            # Open a CSV per object (from your table)
            for it in props.items:
                root = it.created_object
                if not root:
                    continue
                safe_name = _sanitize_filename(root.name)
                path = os.path.join(run_dir, f"{safe_name}.csv")
                fh = open(path, "w", newline="")
                files[it.name] = fh
                w = csv.writer(fh)
                w.writerow(header)
                writers[it.name] = w

            # Iterate frames and write a single row per frame per object
            for f in range(start, end + 1):
                scn.frame_set(f)

                for it in props.items:
                    root = it.created_object
                    if not root:
                        continue
                    w = writers.get(it.name)
                    if not w:
                        continue

                    mesh_objs = _gather_mesh_objects(root)
                    tris = _collect_world_triangles(mesh_objs, deps)

                    # Per-object, per-frame seed → repeatable if seed_base set
                    seed = None if seed_base is None else (hash((seed_base, it.name, f)) & 0xFFFFFFFF)
                    pts = _sample_surface_points(tris, N, seed=seed)

                    # Flatten to one row
                    flat = []
                    for p in pts:
                        flat.extend([p.x, p.y, p.z])
                    w.writerow([f] + flat)

        except Exception as e:
            # Close any opened files on error
            for fh in files.values():
                try: fh.close()
                except Exception: pass
            self.report({'ERROR'}, f"Export failed: {e}")
            return {'CANCELLED'}

        # Close files cleanly
        for fh in files.values():
            fh.close()

        # Reset timeline to a clean starting point
        scn.frame_start = 1
        scn.frame_end   = end
        scn.frame_current = 1

        self.report({'INFO'}, f"Exported per-object, per-frame rows to: {run_dir}. Timeline reset to frame 1.")
        return {'FINISHED'}


class MYADDON_OT_render_video(Operator):
    bl_idname = "myaddon.render_video"
    bl_label = "Render Video"
    bl_description = "Render animation to MP4 using bake settings; resets timeline to frame 1 after render"

    def execute(self, context):
        scn = context.scene
        props = scn.my_addon_props

        outdir = bpy.path.abspath(props.output_dir)
        os.makedirs(outdir, exist_ok=True)
        base = os.path.join(outdir, props.file_base)

        fps = props.bake_fps
        total_frames = props.bake_seconds * fps
        start = 1
        end   = start + total_frames

        scn.frame_start = start
        scn.frame_end   = end
        scn.frame_current = start
        scn.render.fps  = fps
        scn.render.image_settings.file_format = 'FFMPEG'
        scn.render.ffmpeg.format = 'MPEG4'
        scn.render.ffmpeg.codec  = 'H264'
        scn.render.ffmpeg.constant_rate_factor = 'MEDIUM'
        scn.render.ffmpeg.gopsize = 12
        scn.render.filepath = base + ".mp4"

        bpy.ops.render.render(animation=True)

        scn.frame_start = 1
        scn.frame_end   = end
        scn.frame_current = 1
        self.report({'INFO'}, f"Rendered: {scn.render.filepath}. Timeline reset to frame 1.")
        return {'FINISHED'}


class MYADDON_OT_item_remove(Operator):
    bl_idname = "myaddon.item_remove"
    bl_label = "Remove Row"
    bl_description = "Remove selected row (does not delete the scene object)"

    @classmethod
    def poll(cls, context):
        p = context.scene.my_addon_props
        return len(p.items) > 0 and 0 <= p.active_index < len(p.items)

    def execute(self, context):
        p = context.scene.my_addon_props
        p.items.remove(p.active_index)
        p.active_index = max(0, min(p.active_index, len(p.items) - 1))
        return {'FINISHED'}


class MYADDON_OT_item_select_in_viewport(Operator):
    bl_idname = "myaddon.item_select_in_viewport"
    bl_label = "Select Object"
    bl_description = "Select the scene object for the active row"

    @classmethod
    def poll(cls, context):
        p = context.scene.my_addon_props
        return len(p.items) > 0 and 0 <= p.active_index < len(p.items)

    def execute(self, context):
        p = context.scene.my_addon_props
        it = p.items[p.active_index]
        ob = it.created_object
        if ob:
            bpy.ops.object.select_all(action='DESELECT')
            ob.select_set(True)
            context.view_layer.objects.active = ob
        return {'FINISHED'}


class MYADDON_UL_items(UIList):
    bl_idname = "MYADDON_UL_items"
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        icon_id = 'OUTLINER_OB_EMPTY' if (item.created_object and item.created_object.type == 'EMPTY') else 'OUTLINER_OB_MESH'
        split = layout.split(factor=0.45)
        left = split.row(align=True)
        left.prop(item, "name", text="", emboss=False, icon=icon_id)
        right = split.row(align=True)
        right.label(text=f"{item.asset_name}")
        if item.created_object:
            right.label(text=f"• {item.created_object.name}", icon='OBJECT_DATA')

# ---------------------------------------------------------
# Panel (Simple, first block = placement-at-import)
# ---------------------------------------------------------

class TEST_PT_DataGenPanel(Panel):
    bl_label = "Data Generation For MERS Lab"
    bl_idname = "PT_DataGenPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Data Generation"

    def draw(self, context):
        layout = self.layout
        layout.scale_y = 1.06
        props = context.scene.my_addon_props

        # --- Quick Add (single, easy block) ---
        box = layout.box()
        row = box.row(); row.label(text="Object to insert:")
        row = box.row(); row.prop(props, "my_enum", text="")

        place = box.box(); place.label(text="Placement (on Import)")
        place.prop(props, "place_rel_mode", text="Relative")
        if props.place_rel_mode == "WORLD":
            draw_vec3_inline(place, props, "location", title="Location (World)")
        else:
            place.prop(props, "place_reference", text="Reference")
            place.prop(props, "place_on_top", text="Snap On Top")
            if props.place_on_top:
                row = place.row(align=True)
                row.prop(props, "place_align_xy", text="Center XY")
                row.prop(props, "place_margin", text="Top Margin")
            draw_vec3_inline(place, props, "place_offset", title="Extra Offset")

        motion = box.box(); motion.label(text="Initial Motion (for Bake)")
        motion.prop(props, "rel_frame", text="Relative To")
        if props.rel_frame == "OBJECT":
            motion.prop(props, "reference_object", text="Motion Reference")
        draw_vec3_inline(motion, props, "velocity", title="Velocity (m/s)")

        row = box.row(); row.operator("myaddon.import_selected", icon='IMPORT', text="Add Selected Asset")

        layout.separator()

        # Table
        row = layout.row(); row.label(text="Active Objects")
        row = layout.row()
        row.template_list("MYADDON_UL_items", "", props, "items", props, "active_index", rows=6)
        col = row.column(align=True)
        col.operator("myaddon.item_select_in_viewport", text="", icon='RESTRICT_SELECT_OFF')
        col.operator("myaddon.item_remove", text="", icon='X')

        if props.items and 0 <= props.active_index < len(props.items):
            it = props.items[props.active_index]
            box2 = layout.box(); box2.label(text="Edit Selected")
            box2.prop(it, "name", text="Name")
            box2.label(text=f"Asset: {it.asset_name}")
            draw_vec3_inline(box2, it, "location", title="Location")
            draw_vec3_inline(box2, it, "velocity", title="Velocity (m/s)")
            rowrf = box2.row(align=True); rowrf.prop(it, "rel_frame", text="Relative To")
            if it.rel_frame == "OBJECT":
                box2.prop(it, "reference_object", text="Reference Object")
            if it.created_object:
                rowob = box2.row(); rowob.label(text=f"Linked Object: {it.created_object.name}", icon='OBJECT_DATA')

        layout.separator()

        # Recording / Export
        rec = layout.box(); rec.label(text="Recording / Export (resets to frame 1 after export/render)")
        row = rec.row(align=True); row.prop(props, "bake_fps"); row.prop(props, "bake_seconds")
        row = rec.row(align=True); row.prop(props, "pc_width"); row.prop(props, "pc_height")
        row = rec.row(align=True); row.prop(props, "output_dir"); row.prop(props, "file_base")
        row = rec.row(align=True); row.prop(props, "use_active_camera")
        if not props.use_active_camera:
            row = rec.row(align=True); row.prop(props, "camera_override")
        row = rec.row(align=True)
        row.operator("myaddon.bake_motion", icon='ANIM')
        row.operator("myaddon.export_pointclouds", icon='EXPORT')
        row = rec.row(align=True); row.prop(props, "make_video"); row.prop(props, "video_kind")
        if props.make_video:
            row = rec.row(align=True); row.operator("myaddon.render_video", icon='RENDER_ANIMATION')

# ---------------------------------------------------------
# Register
# ---------------------------------------------------------

classes = (
    MyItem,
    MyAddonProperties,
    MYADDON_OT_import_selected,
    MYADDON_OT_bake_motion,
    MYADDON_OT_export_pointclouds,
    MYADDON_OT_render_video,
    MYADDON_OT_item_remove,
    MYADDON_OT_item_select_in_viewport,
    MYADDON_UL_items,
    TEST_PT_DataGenPanel,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.my_addon_props = PointerProperty(type=MyAddonProperties)


def unregister():
    del bpy.types.Scene.my_addon_props
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
