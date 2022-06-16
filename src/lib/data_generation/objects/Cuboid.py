import numpy as np
from trimesh.creation import box
import sys, os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from data_generation import Base
from utils.transform import create_homog_matrix, create_rot_mat_axisAlign

# Basic Configuration

# Customized parameters

#
#             pt1_ _ _ _ _ _ _ _ _pt2
#              /|      z    y     /|
#             / |      |   /     / |
#         pt3/_ | _ _ _|  /__pt4/  |
#           |   |      | /     |   |
#           |   |      |/______|___|_____x
#           |  pt5_ _ _|_ _ _ _|_ _|pt6
#           |  /       |       |  /
#           | /        |       | /
#        pt7|/_ _ _ _ _ _ _ _ _|/pt8

# x1, y1, z1 = (-x/2, y/2, z/2)  # | => pt1
# x2, y2, z2 = (x/2, y/2, z/2)  # | => pt2
# x3, y3, z3 = (-x/2, -y/2, z/2)  # | => pt3
# x4, y4, z4 = (x/2, -y/2, z/2)  # | => pt4
# x5, y5, z5 = (-x/2, y/2, -z/2)  # | => pt5
# x6, y6, z6 = (x/2, y/2, -z/2)  # | => pt6
# x7, y7, z7 = (-x/2, -y/2, -z/2)  # | => pt7
# x8, y8, z8 = (x/2, -y/2, -z/2)  # | => pt8

class Cuboid(Base):
    """The Cuboid class
    The cuboid frame origin is defined at the center of the cuboid, and the axes perpendicular to the surfaces
    """
    def __init__(self, x_size, y_size, z_size, color=np.random.choice(range(256), size=3).astype(np.uint8), 
                pose=None, trl_sample_num=5) -> None:
        """Generate a Cuboid instance

        Args:
            x_size (float): size in x direction in m
            y_size (float): size in y direction in m
            z_size (float): size in z direction in m
            color (array (3,)): The color of the object. Defaults to random color
            pose (array (4,4), optional): The initial 4-by-4 homogenous transformation matrix of cuboid-to-world. Defaults to None (align with the world frame)
        """

        # store the size
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size

        # create the instance
        super().__init__(color=color, pose=pose, trl_sample_num=trl_sample_num)

        self.type = "cuboid"

        # print('mesh faces: ', self.obj_mesh.faces) # mesh faces are triangles

    
    def generate_mesh(self):
        obj_mesh = box(
            extents=[self.x_size, self.y_size, self.z_size]
        )

        return obj_mesh

    def generate_grasp_family(self):
        """The cuboid grasp family in the object frame.
        The Open width is set to be 1.2 times the corresponding object dimension

        Returns:
            grasp_poses (array, (N, 4, 4)): The homogeneous grasp poses in the object frame
            open_widths (array, (N,)): The open widths
        """
        sample_num = self.trl_sample_num
        sink_in_size = 0.02 # the gripper should sink in 2 cm

        grasp_poses = []
        open_widths = []
        # The grasp could point to six directions in the object frame
        for x in [1, -1, 2, -2, 3, -3]:
            # z (i.e. the gripper closing direction) could point to both the other directions and is origin-symmetrical
            # so just point to the positive direction
            z_options = [1, 2, 3]
            z_options.remove(abs(x))
            for z in z_options:
                # use the right hand rule (x and -z) to determine the y directions
                x_vec = np.array([0, 0, 0])
                z_vec = np.array([0, 0, 0])
                x_vec[abs(x)-1] = x/abs(x)
                z_vec[abs(z)-1] = z/abs(z)
                
                y_vec = np.cross(x_vec, -z_vec)
                y_options = [1, 2, 3]
                y_options.remove(abs(x))
                y_options.remove(abs(z))
                y = y_options[0] * int(np.sum(y_vec))

                # create the rotation matrix
                rot_mat = create_rot_mat_axisAlign([x, y, z])

                # create the translation vector. It involves sampling over the gripper y direction
                for i in range(sample_num):
                    trans_vec = [0, 0, 0]
                    trans_vec[abs(x) - 1] = - x/abs(x) * (self._fetch_direction_size(x) / 2 - sink_in_size)
                    trans_vec[abs(y) - 1] = (- 1/2 + (i+1) / (sample_num + 1)) * self._fetch_direction_size(y)

                    pose = create_homog_matrix(rot_mat, trans_vec)
                    grasp_poses.append(pose)

                    open_widths.append(1.2 * self._fetch_direction_size(z))

        return np.stack(grasp_poses, axis=0), np.array(open_widths)

    def _fetch_direction_size(self, direction_idx):
        """Fetch the cuboid size along a direction.
        Denote the correspondence: 1 - x; 2 - y; 3 - z

        Args:
            direction_idx (int): 1/2/3, meaning x/y/z
        """
        direction_idx = abs(direction_idx)
        assert direction_idx in [1, 2, 3]
        if direction_idx == 1:
            return self.x_size 
        elif direction_idx == 2:
            return self.y_size
        elif direction_idx == 3:
            return self.z_size
        
    def get_obj_type(self):
        return "cuboid"
    
    def get_obj_dims(self):
        """ The cuboid size is quantified as a (3, ) array 
        storing the x_size, y_size, z_size
        """
        return np.array([self.x_size, self.y_size, self.z_size])
    
    @staticmethod
    def construct_obj_info(obj_dims, obj_pose, trl_sample_num=5, **kwargs):
        x_size = obj_dims[0]
        y_size = obj_dims[1]
        z_size = obj_dims[2]
        return Cuboid(x_size, y_size, z_size, pose=obj_pose,trl_sample_num=trl_sample_num, **kwargs)


""" PSCNN's code
def generate_cuboid(port_num,x_input,y_input,z_input):
    
    scaleReduceFactor=100
    x=x_input/scaleReduceFactor
    y=y_input/scaleReduceFactor
    z=z_input/scaleReduceFactor

    x1, y1, z1 = (-x/2, y/2, z/2)  # | => pt1
    x2, y2, z2 = (x/2, y/2, z/2)  # | => pt2
    x3, y3, z3 = (-x/2, -y/2, z/2)  # | => pt3
    x4, y4, z4 = (x/2, -y/2, z/2)  # | => pt4
    x5, y5, z5 = (-x/2, y/2, -z/2)  # | => pt5
    x6, y6, z6 = (x/2, y/2, -z/2)  # | => pt6
    x7, y7, z7 = (-x/2, -y/2, -z/2)  # | => pt7
    x8, y8, z8 = (x/2, -y/2, -z/2)  # | => pt8

    mlab.options.offscreen = True
    figure = mlab.gcf()
    mlab.clf()
    figure.scene.disable_render = True
    
    s=mayavi.mlab.mesh([[x1, x2, x6], [x3, x4, x8], [x7, x8, x8],[x8, x6, x2], [x7, x5, x1], [x3, x1, x1]],
                    [[y1, y2, y6], [y3, y4, y8], [y7, y8, y8],[y8, y6, y2], [y7, y5, y1], [y3, y1, y1]],
                    [[z1, z2, z6], [z3, z4, z8], [z7, z8, z8],[z8, z6, z2], [z7, z5, z1], [z3, z1, z1]],
                    )
    
    # mayavi.mlab.close()
    # Show
    # mayavi.mlab.show()

    actor = s.actor.actors[0]
    polydata = tvtk.to_vtk(actor.mapper.input)

    ## Save to stl
    stlWriter = vtk.vtkSTLWriter()
    # Set the file name
    stlWriter.SetFileName('objects/Object_Generation_'+port_num+'.stl')
    stlWriter.SetInputData(polydata)
    stlWriter.Write()
    # print('Done')

    scene=load('objects/Object_Generation_'+port_num+'.stl')
    export(scene, 'objects/Object_Generation_'+port_num+'.obj',file_type='obj')
    os.remove('objects/Object_Generation_'+port_num+'.obj.mtl')

    kp_3d = np.array([[-x/2, y/2, z/2], [x/2, y/2, z/2], [-x/2, -y/2, z/2], [x/2, -y/2, z/2], \
             [-x/2, y/2, -z/2], [x/2, y/2, -z/2], [-x/2, -y/2, -z/2], [x/2, -y/2, -z/2]])

    return kp_3d

"""

if __name__=="__main__":
    cuboid = Cuboid(0.04, 0.08, 0.16)
    cuboid.vis(world_frame=True, obj_frame=True, gripper_frame=True)