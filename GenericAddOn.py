import bpy

class Test_Panel(bpy.types.Panel):
    bl_label = "Test Panel"
    bl_idname = "PT_TestPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NewTab"
    
    def draw(self, context):
        layout = self.layout
        obj = context.object
        layout.scale_y = 1.25
        
        row = layout.row()
        row.label(text = "oooOOoOOo Spooky", icon = 'GHOST_ENABLED')
        
        row = layout.row()
        row.operator("mesh.primitive_cube_add")
        
        row = layout.row()
        row.operator("mesh.primitive_ico_sphere_add")
        
        row = layout.row()
        row.operator("object.text_add")
        
        row = layout.row()
        row.label(text = "Scaling and Sizing", icon= "ERROR")
        row = layout.row()
        row.operator("transform.resize")

        column = layout.column()
        column.prop(obj, "scale")
        
        
        
        
        

def register():
    bpy.utils.register_class(Test_Panel)
    
    
def unregister():
    bpy.utils.unregister_class(Test_Panel)
    
    

if __name__ == "__main__":
    register()