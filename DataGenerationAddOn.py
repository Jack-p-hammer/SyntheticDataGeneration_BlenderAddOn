import bpy
from bpy.props import (
    EnumProperty, PointerProperty, FloatVectorProperty
)
from bpy.types import Panel, PropertyGroup, Operator
from pathlib import Path

# --- CONFIG ---
ASSET_DIR = Path(r"C:/Users/jonat/Documents/Course Work/Current Courses/UROP/SyntheticDataGeneration/CustomAssets")

# --- Dynamic enum callback (refreshes when the panel is opened) ---
def enum_assets(self, context):
    items = []
    if ASSET_DIR.exists():
        for p in ASSET_DIR.glob("*.blend"):
            name = p.stem
            items.append((name, name, f"Import {name}"))
    if not items:
        items = [("NONE", "No .blend assets found", "Put .blend files in ASSET_DIR")]
    return items


class MyAddonProperties(PropertyGroup):
    my_enum: EnumProperty(
        name="Objects",
        description="Choose an object from the list",
        items=enum_assets,  # dynamic
    )
    location: FloatVectorProperty(
        name="Location",
        subtype='TRANSLATION',
        size=3,
        default=(0.0, 0.0, 0.0),
    )
    velocity: FloatVectorProperty(
        name="Velocity (m/s)",
        subtype='VELOCITY',
        size=3,
        default=(0.0, 0.0, 0.0),
    )

    # Relative motion controls
    rel_frame: EnumProperty(
        name="Relative To",
        items=[
            ("WORLD", "World", "Velocity in world frame"),
            ("OBJECT", "Object", "Velocity relative to a reference object"),
        ],
        default="WORLD",
    )
    reference_object: PointerProperty(
        name="Reference Object",
        type=bpy.types.Object,
        description="Existing scene object to use as reference when rel frame = Object",
    )


def draw_vec3_column_inline(layout, data, prop, title="Vector"):
    box = layout.box()
    box.use_property_split = False
    box.use_property_decorate = False

    split = box.split(factor=0.33)
    col_l = split.column(align=True)
    col_r = split.column(align=True)

    # First row: title inline with X
    col_l.label(text=title)
    row = col_r.row(align=True)
    row.prop(data, prop, index=0, text="X")

    # Keep alignment for Y and Z
    col_l.label(text="")
    col_r.prop(data, prop, index=1, text="Y")

    col_l.label(text="")
    col_r.prop(data, prop, index=2, text="Z")


# --- Operator: import selected asset and place at location ---
class MYADDON_OT_import_selected(Operator):
    bl_idname = "myaddon.import_selected"
    bl_label = "Add Selected Asset"
    bl_description = "Append the chosen .blend asset into the scene and place it at Location"

    def execute(self, context):
        props = context.scene.my_addon_props
        name = props.my_enum

        if name == "NONE":
            self.report({'WARNING'}, "No assets to import. Put .blend files in ASSET_DIR.")
            return {'CANCELLED'}

        libpath = ASSET_DIR / f"{name}.blend"
        if not libpath.exists():
            self.report({'ERROR'}, f"Missing file: {libpath}")
            return {'CANCELLED'}

        appended_any = False
        try:
            # Append collections if present, else objects
            with bpy.data.libraries.load(str(libpath), link=False) as (data_from, data_to):
                if data_from.collections:
                    data_to.collections = list(data_from.collections)
                elif data_from.objects:
                    data_to.objects = list(data_from.objects)
                else:
                    self.report({'WARNING'}, f"No Collections or Objects found in {libpath.name}")
                    return {'CANCELLED'}

            # Link appended data to the current scene and place them
            if hasattr(data_to, "collections") and data_to.collections:
                for coll in data_to.collections:
                    # Link collection if not already linked
                    if coll.name not in [c.name for c in context.scene.collection.children]:
                        context.scene.collection.children.link(coll)
                    for ob in coll.objects:
                        ob.location = props.location
                appended_any = True

            if hasattr(data_to, "objects") and data_to.objects:
                for ob in data_to.objects:
                    # May already be linked via collection; if not, link directly
                    if ob.name not in [o.name for o in context.scene.collection.objects]:
                        try:
                            context.scene.collection.objects.link(ob)
                        except Exception:
                            pass
                    ob.location = props.location
                appended_any = True

        except Exception as e:
            self.report({'ERROR'}, f"Append failed: {e}")
            return {'CANCELLED'}

        if not appended_any:
            self.report({'WARNING'}, "Nothing was appended from the file.")
            return {'CANCELLED'}

        loc = tuple(round(v, 3) for v in props.location)
        self.report({'INFO'}, f"Appended from {libpath.name} at {loc}")
        return {'FINISHED'}


class TEST_PT_DataGenPanel(Panel):
    bl_label = "Data Generation For MERS Lab"
    bl_idname = "PT_DataGenPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Data Generation"

    def draw(self, context):
        layout = self.layout
        layout.scale_y = 1.25

        props = context.scene.my_addon_props

        # Asset dropdown
        row = layout.row()
        row.label(text="Object to insert:")
        row = layout.row()
        row.prop(props, "my_enum", text="")  # dynamic enum

        # Location / Velocity
        draw_vec3_column_inline(layout, props, "location", title="Location")
        draw_vec3_column_inline(layout, props, "velocity", title="Velocity (m/s)")

        # --- IMPLEMENTED: Relative motion controls (World vs Object) ---
        box = layout.box()
        box.label(text="Relative Motion")
        box.prop(props, "rel_frame", text="Relative To")
        if props.rel_frame == "OBJECT":
            box.prop(props, "reference_object", text="Reference Object")

        # --- IMPLEMENTED: Add the object to the world ---
        row = layout.row()
        row.operator("myaddon.import_selected", text="Add Selected Asset", icon='IMPORT')


# --- Register ---
classes = (
    MyAddonProperties,
    MYADDON_OT_import_selected,
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