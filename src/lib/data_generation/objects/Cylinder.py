import numpy as np
from numpy import pi, sin, cos
from pyassimp import *
# Basic Configuration
import os


import sys, os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import trimesh
from data_generation import Base
from utils.transform import create_homog_matrix, create_rot_mat_axisAlign

class Cylinder(Base):
    """The Cylinder class
    The inner and outer radius ratio if fixed to be 1.15 following the PSCNN
    The rotation axis of the cylinder aligns with the z axis
    """
    def __init__(self, r_in, height, mode="cylinder", color=np.random.choice(range(256), size=3).astype(np.uint8), \
        pose=None, trl_sample_num=5, ang_sample_num=11) -> None:
        """Generate a Cylinder instance

        Args:
            r_in (float): The inner radius in m
            height (float): The height of the cylinder in m
            mode (str): "cylinder" or "stick" or "ring". Will influence on the grasp family generation
            color (array (3,)): The color of the object. Defaults to random color
            pose (array (4,4), optional): The initial 4-by-4 homogenous transformation matrix of cuboid-to-world. Defaults to None (align with the world frame)
        """

        # set the size cm to m
        self.r_in = r_in
        self.r_out = self.r_in * 1.15
        self.height = height

        # store the mode
        self.mode = mode

        # create the instance
        super().__init__(pose=pose, color=color, trl_sample_num=trl_sample_num, ang_sample_num=ang_sample_num)
    
    def generate_mesh(self):
        """The PSCNN code (gneerate_mesh_vertices) seems not creating the inner surface, making the mesh incomplete.
        So temporarily overwrite it.
        """
        if self.mode == "stick":
            obj_mesh = trimesh.creation.cylinder(self.r_in, self.height)
        else:
            obj_mesh = trimesh.creation.annulus(self.r_in, self.r_out, self.height)
        
        return obj_mesh
    
    def generate_grasp_family(self):

        # parameters
        ang_sample_num = self.ang_sample_num
        trl_sample_num = self.trl_sample_num
        sink_in_dist = 0.005

        # result storage
        grasp_poses = []
        open_widths = []

        # grasp family from top & bottom
        if self.mode != "stick":
            grasp_poses_TB, open_width_TB = self._generate_grasp_top_bottom(ang_sample_num=ang_sample_num, sink_in_dist=sink_in_dist)
            grasp_poses = grasp_poses + grasp_poses_TB
            open_widths = open_widths + [open_width_TB] * len(grasp_poses_TB)

        # grasp family from the side
        if self.mode != "ring":
            grasp_poses_side, open_width_side = self._generate_grasp_side(ang_sample_num=ang_sample_num, trl_sample_num=trl_sample_num)
            grasp_poses = grasp_poses + grasp_poses_side
            open_widths = open_widths + [open_width_side] * len(grasp_poses_side)

        return np.array(grasp_poses), np.array(open_widths)
    
    def _generate_grasp_top_bottom(self, ang_sample_num=11, sink_in_dist=0.02):
        """The grasp family from top and bottom
        Free parameter is the rotation along the z axis (height direction)

        Return:
            grasp_poses (list).  A list of 4-by-4 grasp poses
            open_width (float).  The open width. All the grasp poses in this category share the same open width
        """
        open_width = 2 * (self.r_out - self.r_in)

        # transformation to the top & bottom center
        R_top = create_rot_mat_axisAlign([-3, 2, 1])
        T_top = [0, 0, self.height/2 - sink_in_dist]
        trf_top = create_homog_matrix(R_top, T_top)
        R_bottom = create_rot_mat_axisAlign([3, -2, 1])
        T_bottom = [0, 0, - self.height/2 + sink_in_dist]
        trf_bottom = create_homog_matrix(R_bottom, T_bottom)
        
        # sample the rotational free parameter
        grasp_poses = []
        angle_samples = np.linspace(0, 2*pi, ang_sample_num)
        for trf in [trf_top, trf_bottom]:
            # first translate then rotate along z axis
            trl = [(self.r_in + self.r_out)/2, 0, 0]
            trl_homog = create_homog_matrix(T_vec=trl)
            for theta in angle_samples:
                rot =   [[cos(theta), -sin(theta), 0],
                        [sin(theta), cos(theta), 0],
                        [0, 0, 1]]
                rot_homog = create_homog_matrix(R_mat=rot) 
                grasp_poses.append(rot_homog@trl_homog@trf)

        return grasp_poses, open_width
    
    def _generate_grasp_side(self, ang_sample_num=11, trl_sample_num=5):
        """The grasp family on the side.
        Free parameters involve the rotation and translation along the z axis (height direction)

        Return:
            grasp_poses (list).  A list of 4-by-4 grasp poses
            open_width (float).  The open width. All the grasp poses in this category share the same open width
        """
        open_width = 1.2 * (2 * self.r_out)

        # transform to side 
        R = create_rot_mat_axisAlign([1, -3, 2])
        trf = create_homog_matrix(R)

        # sample the free parameter
        grasp_poses = []
        angle_samples = np.linspace(0, 2*pi, ang_sample_num)
        trl_samples = np.linspace(-self.height/2, self.height/2, trl_sample_num+2)
        trl_samples = trl_samples[1:-1] # remove the first & last element
        for theta in angle_samples:
            rot =   [[cos(theta), -sin(theta), 0],
                    [sin(theta), cos(theta), 0],
                    [0, 0, 1]]
            for trl in trl_samples:
                trl_vec = [0, 0, trl]
                trf_sample = create_homog_matrix(rot, trl_vec)
                grasp_poses.append(trf_sample @ trf)
        
        return grasp_poses, open_width

    
    def get_obj_type(self):
        return self.mode


    def get_obj_dims(self):
        """
        (r_in, height)
        """
        return np.array([self.r_in, self.height])
    

    @staticmethod
    def construct_obj_info(obj_dims, obj_pose, mode, trl_sample_num=5, rot_sample_num=11, **kwargs):
        r_in = obj_dims[0]
        h = obj_dims[1]

        assert mode in ["cylinder", "stick", "ring"]

        return Cylinder(r_in=r_in, height=h, mode=mode, pose=obj_pose, 
                        ang_sample_num=rot_sample_num, trl_sample_num=trl_sample_num, **kwargs)


"""The PSCNN code
def generate_cylinder(port_num,rin_input,height_input):
    # Customized parameters
    
    scaleReduceFactor=100

    rin = rin_input/scaleReduceFactor
    rout = rin * 1.15
    height = height_input/scaleReduceFactor

    # size=1
    # # Cylinder
    # rin = (2.2 + size * 0.8)/scaleReduceFactor
    # rout = rin*1.15
    # height = rin * 2.45

    # Ring
    # rin = (size*0.35+1.5)/scaleReduceFactor
    # rout = rin * 1.25
    # height = rin * 0.625

    # # Stick
    # rin = 0
    # rout =1.25/scaleReduceFactor
    # height = (4 + size*1.5)/scaleReduceFactor

    alpha = np.arange(0,2*pi+2*pi/100,2*pi/100)
    alpha=alpha.reshape(1,alpha.shape[0])
    xin = rin * cos(alpha)
    xout = rout * cos(alpha)
    yin = rin * sin(alpha)
    yout = rout * sin(alpha)
    h1 = -height/2
    h2 = height/2

    # Basic shape
    unit_cylinder_x=np.repeat(cos(alpha),2,axis=0)
    unit_cylinder_y=np.repeat(sin(alpha),2,axis=0)
    unit_cylinder_z=np.repeat(np.array([[0],[1]]),xout.shape[1],axis=1)

    # Build up faces
    cylinder_face_x=np.concatenate((unit_cylinder_x*rout,unit_cylinder_x*rin),axis=1)
    cylinder_face_y=np.concatenate((unit_cylinder_y*rout,unit_cylinder_y*rin),axis=1)
    cylinder_face_z=np.concatenate((unit_cylinder_z*(h2-h1)+h1,unit_cylinder_z*(h2-h1)+h1),axis=1)

    up_face_x=np.concatenate((xin,xout),axis=0)
    up_face_y=np.concatenate((yin,yout),axis=0)
    up_face_z=h1*np.ones((2,xout.shape[1]))

    down_face_x=np.concatenate((xin,xout),axis=0)
    down_face_y=np.concatenate((yin,yout),axis=0)
    down_face_z=h2*np.ones((2,xout.shape[1]))

    X=np.concatenate((cylinder_face_x,up_face_x,down_face_x),axis=1)
    Y=np.concatenate((cylinder_face_y,up_face_y,down_face_y),axis=1)
    Z=np.concatenate((cylinder_face_z,up_face_z,down_face_z),axis=1)

    
    mlab.options.offscreen = True
    figure = mlab.gcf()
    mlab.clf()
    figure.scene.disable_render = True
    s = mlab.mesh(X, Y, Z)
#     mayavi.mlab.close()
    # Show
    # mlab.show()



    # Export polydata
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

    kp_3d=np.array([[-rout,rout,h2],[rout,rout,h2],[-rout,-rout,h2],[rout,-rout,h2],\
           [-rout,rout,h1],[rout,rout,h1],[-rout,-rout,h1],[rout,-rout,h1]])

    return kp_3d
"""

if __name__ == "__main__":
    # cylinder = Cylinder(0.05, 0.08)
    # cylinder.vis(True, False, False, False)

    # ring = Cylinder(0.03, 0.012, mode="ring")
    # ring.vis(True, False, False, False)

    stick = Cylinder(0.01, 0.10, mode="stick", color=[0.5, 0.5, 0.5, 0.5])
    # create a random pose
    stick.set_pose(create_homog_matrix(
        R_mat=np.eye(3),
        T_vec=np.array([0.2, 0.1, 0])
    ))
    stick.vis(False, True, True, False)
