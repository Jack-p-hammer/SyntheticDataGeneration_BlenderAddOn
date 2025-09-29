# ==== Synthetic Data Add-on: Assets + Bake Motion + Depth Point Clouds + Video ====
# Works on Blender 3.x / 4.x
import bpy
from bpy.props import (
    EnumProperty, PointerProperty, FloatVectorProperty, CollectionProperty,
    IntProperty, StringProperty, BoolProperty
)
from bpy.types import Panel, PropertyGroup, Operator, UIList
from pathlib import Path
from mathutils import Vector
import csv
import os
import math

bl_info = {
    "name": "MERS Lab – Synthetic Data Builder",
    "author": "Jack + ChatGPT",
    "version": (0, 3, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Data Generation",
    "description": "Import assets, bake simple motions, export depth-style point clouds, render video.",
    "category": "Object",
}

# --- CONFIG: change to your asset folder ---
ASSET_DIR = Path(r"C:/Users/jonat/Documents/Course Work/Current Courses/UROP/SyntheticDataGeneration/CustomAssets")

# --- Dynamic enum: list .blend files in ASSET_DIR ---
def enum_assets(self, context):
    items = []
    if ASSET_DIR.exists():
        for p in sorted(ASSET_DIR.glob("*.blend")):
            name = p.stem
            items.append((name, name, f"Import {name}"))
    if not items:
        items = [("NONE", "No .blend assets found", "Put .blend files in ASSET_DIR")]
    return items

# ----------------------- Helpers -----------------------
def _child_names(coll):
    try:
        return [c.name for c in coll.children]
    except Exception:
        return []

def _object_names(coll):
    try:
        return [o.name for o in coll.objects]
    except Exception:
        return []

def draw_vec3_column_inline(layout, data, prop, title="Vector"):
    box = layout.box()
    box.use_property_split = False
    box.use_property_decorate = False
    split = box.split(factor=0.33)
    col_l = split.column(align=True)
    col_r = split.column(align=True)
    col_l.label(text=title); row = col_r.row(align=True); row.prop(data, prop, index=0, text="X")
    col_l.label(text="");    col_r.prop(data, prop, index=1, text="Y")
    col_l.label(text="");    col_r.prop(data, prop, index=2, text="Z")

def _ensure_camera(context, props):
    cam = context.scene.camera if props.use_active_camera else props.camera_override
    if cam and cam.type == 'CAMERA':
        return cam
    # create a simple camera if none
    bpy.ops.object.camera_add(location=(0.0, -5.0, 2.0), rotation=(math.radians(75), 0, 0))
    context.scene.camera = bpy.context.active_object
    return context.scene.camera

# Ray for pixel (i,j) in WxH using camera frustum
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

# ----------------------- Data Model -----------------------
def _item_location_update(self, context):
    if self.created_object:
        self.created_object.location = Vector(self.location)

def _item_name_update(self, context):
    if self.created_object and self.name:
        self.created_object.name = self.name

class MyItem(PropertyGroup):
    name: StringProperty(
        name="Name", description="Display/Object name", default="",
        update=_item_name_update
    )
    asset_name: StringProperty(name="Asset", description="File/datablock stem", default="")
    location: FloatVectorProperty(
        name="Location", subtype='TRANSLATION', size=3, default=(0,0,0),
        update=_item_location_update
    )
    velocity: FloatVectorProperty(
        name="Velocity (m/s)", subtype='VELOCITY', size=3, default=(0,0,0)
    )
    rel_frame: EnumProperty(
        name="Relative To",
        items=[("WORLD","World","World-frame velocity"),
               ("OBJECT","Object","Velocity relative to a reference object")],
        default="WORLD"
    )
    reference_object: PointerProperty(name="Reference Object", type=bpy.types.Object)
    created_object: PointerProperty(name="Created Object", type=bpy.types.Object)

class MyAddonProperties(PropertyGroup):
    # Quick Add
    my_enum: EnumProperty(name="Objects", description="Choose asset", items=enum_assets)
    location: FloatVectorProperty(name="Location", subtype='TRANSLATION', size=3, default=(0,0,0))
    velocity: FloatVectorProperty(name="Velocity (m/s)", subtype='VELOCITY', size=3, default=(0,0,0))
    rel_frame: EnumProperty(name="Relative To", items=[("WORLD","World",""),("OBJECT","Object","")], default="WORLD")
    reference_object: PointerProperty(name="Reference Object", type=bpy.types.Object)

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

# ----------------------- Import Operator -----------------------
class MYADDON_OT_import_selected(Operator):
    bl_idname = "myaddon.import_selected"
    bl_label   = "Add Selected Asset"
    bl_description = "Append datablock named like the dropdown, place at Location, add to table"

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
                # Create a controller Empty to move the whole asset group
                if coll and coll.objects:
                    ctrl = bpy.data.objects.new(f"{name}_ctrl", None)
                    ctrl.empty_display_type = 'PLAIN_AXES'
                    context.scene.collection.objects.link(ctrl)
                    ctrl.location = Vector(props.location)
                    for ob in coll.objects:
                        ob.parent = ctrl
                        ob.matrix_parent_inverse = ctrl.matrix_world.inverted()
                    created = ctrl
            else:
                ob = bpy.data.objects.get(name)
                if ob:
                    if ob.name not in _object_names(context.scene.collection):
                        context.scene.collection.objects.link(ob)
                    ob.location = Vector(props.location)
                    created = ob

            bpy.context.view_layer.update()
            if created is None:
                self.report({'WARNING'}, f"Nothing linked from {libpath.name}")
                return {'CANCELLED'}

            row = props.items.add()
            row.asset_name = name
            row.created_object = created
            row.name = created.name
            row.location = created.location[:]  # sync
            row.velocity = props.velocity
            row.rel_frame = props.rel_frame
            row.reference_object = props.reference_object
            props.active_index = len(props.items) - 1

        except Exception as e:
            self.report({'ERROR'}, f"Append failed: {e}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Added '{name}' @ {tuple(round(v,3) for v in props.location)}")
        return {'FINISHED'}

# ----------------------- Bake Motion -----------------------
class MYADDON_OT_bake_motion(bpy.types.Operator):
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

        # Make the range visible & start at 1
        scn.frame_start = start
        scn.frame_end   = end
        scn.frame_current = start

        # Capture bases at frame 1
        scn.frame_set(start)
        base = {}
        ref0 = {}
        for it in props.items:
            ob = it.created_object
            if not ob:
                continue
            base[it.name] = ob.location.copy()
            if it.rel_frame == "OBJECT" and it.reference_object:
                ref0[it.name] = it.reference_object.matrix_world.translation.copy()
            else:
                ref0[it.name] = Vector((0,0,0))

        # Key each frame
        for f in range(start, end + 1):
            scn.frame_set(f)
            t = (f - start) / fps
            for it in props.items:
                ob = it.created_object
                if not ob:
                    continue
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


# ----------------------- Depth Point Cloud → ONE CSV -----------------------
class MYADDON_OT_export_pointclouds(bpy.types.Operator):
    bl_idname = "myaddon.export_pointclouds"
    bl_label = "Export Point Clouds (One CSV)"
    bl_description = "Ray-cast from camera and write a single CSV with all frames; resets timeline to frame 1 after export"

    def execute(self, context):
        scn = context.scene
        props = scn.my_addon_props

        cam = context.scene.camera if props.use_active_camera else props.camera_override
        if not cam or cam.type != 'CAMERA':
            self.report({'ERROR'}, "No valid camera found.")
            return {'CANCELLED'}

        fps = props.bake_fps
        total_frames = props.bake_seconds * fps
        start = 1
        end   = start + total_frames

        W, H = props.pc_width, props.pc_height
        outdir = bpy.path.abspath(props.output_dir)
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, f"{props.file_base}.csv")

        # object_id map from table order
        id_map = {it.created_object.name: idx for idx, it in enumerate(props.items) if it.created_object}

        deps = context.evaluated_depsgraph_get()
        clip_end = cam.data.clip_end

        try:
            with open(out_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["frame","object_id","label","u","v","x","y","z","nx","ny","nz"])
                for f in range(start, end + 1):
                    scn.frame_set(f)
                    for v in range(H):
                        for u in range(W):
                            origin, direction = _camera_ray_for_pixel(cam, u, v, W, H)
                            hit, loc, normal, face_idx, ob, mat = scn.ray_cast(deps, origin, direction, distance=clip_end)
                            if not hit or ob is None:
                                continue
                            label = ob.name
                            oid = id_map.get(label, -1)
                            w.writerow([f, oid, label, u, v, loc.x, loc.y, loc.z, normal.x, normal.y, normal.z])
        except Exception as e:
            self.report({'ERROR'}, f"Failed to write CSV: {e}")
            return {'CANCELLED'}

        # 🔁 Reset timeline to a clean starting point
        scn.frame_start = 1
        scn.frame_end   = end
        scn.frame_current = 1

        self.report({'INFO'}, f"Wrote point cloud CSV: {out_path}. Timeline reset to frame 1.")
        return {'FINISHED'}


# ----------------------- Render Video (MP4) -----------------------
class MYADDON_OT_render_video(bpy.types.Operator):
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

        # 🔁 Reset timeline to a clean starting point
        scn.frame_start = 1
        scn.frame_end   = end
        scn.frame_current = 1

        self.report({'INFO'}, f"Rendered: {scn.render.filepath}. Timeline reset to frame 1.")
        return {'FINISHED'}

# ----------------------- Table ops/UI -----------------------
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
        p.active_index = max(0, min(p.active_index, len(p.items)-1))
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

# ----------------------- Panel -----------------------
class TEST_PT_DataGenPanel(Panel):
    bl_label = "Data Generation For MERS Lab"
    bl_idname = "PT_DataGenPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Data Generation"

    def draw(self, context):
        layout = self.layout
        layout.scale_y = 1.08
        props = context.scene.my_addon_props

        # Quick Add
        box = layout.box()
        row = box.row(); row.label(text="Object to insert:")
        row = box.row(); row.prop(props, "my_enum", text="")
        draw_vec3_column_inline(box, props, "location", title="Location")
        draw_vec3_column_inline(box, props, "velocity", title="Velocity (m/s)")
        sub = box.box(); sub.label(text="Relative Motion")
        sub.prop(props, "rel_frame", text="Relative To")
        if props.rel_frame == "OBJECT":
            sub.prop(props, "reference_object", text="Reference Object")
        row = box.row()
        row.operator("myaddon.import_selected", icon='IMPORT', text="Add Selected Asset")

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
            draw_vec3_column_inline(box2, it, "location", title="Location")
            draw_vec3_column_inline(box2, it, "velocity", title="Velocity (m/s)")
            rowrf = box2.row(align=True)
            rowrf.prop(it, "rel_frame", text="Relative To")
            if it.rel_frame == "OBJECT":
                box2.prop(it, "reference_object", text="Reference Object")
            if it.created_object:
                rowob = box2.row()
                rowob.label(text=f"Linked Object: {it.created_object.name}", icon='OBJECT_DATA')

        layout.separator()

        # Recording / Export
        rec = layout.box(); rec.label(text="Recording / Export")
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

# ----------------------- Register -----------------------
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