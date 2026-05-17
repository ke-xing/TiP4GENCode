#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
###
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
###

def render_set(model_path, name, iteration, views, p_views_1, p_views_2, p_views_3, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        depth = render_pkg["depth"]
        gt = view.original_image[0:3, :, :]
        ##########
        scale_nor = depth.max().item()
        depth_nor = depth / scale_nor
        depth_tensor_squeezed = depth_nor.squeeze()  # Remove the channel dimension
        colormap = plt.get_cmap('jet')
        depth_colored = colormap(depth_tensor_squeezed.cpu().numpy())
        depth_colored_rgb = depth_colored[:, :, :3]
        depth_image = Image.fromarray((depth_colored_rgb * 255).astype(np.uint8))
        output_path = os.path.join(depth_path, '{0:05d}'.format(idx) + ".png")
        depth_image.save(output_path)
        ##########
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,frame):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, api_key=None, self_refinement=None, num_prompt=None, max_rounds=None,frame=frame)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        # render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
        # scene.getPerturbationCameras(stage=1), scene.getPerturbationCameras(stage=2), scene.getPerturbationCameras(stage=3), gaussians, pipeline, background)

   
        # train_camera1=scene.getTrainCameras()[0]
        # train_camera2=scene.getTrainCameras()[1]
        # train_camera3=scene.getTrainCameras()[2]
        # train_camera4=scene.getTrainCameras()[3]
        
        # # Create rotation matrices for front, back, left, right facing cameras
 


        # def rotate_matrix(angle_x, angle_y, angle_z):
        #     # Convert angles from degrees to radians
        #     rx = np.radians(angle_x)
        #     ry = np.radians(angle_y)
        #     rz = np.radians(angle_z)

        #     # Rotation matrices for each axis
        #     Rx = np.array([[1, 0, 0],
        #                    [0, np.cos(rx), -np.sin(rx)],
        #                    [0, np.sin(rx), np.cos(rx)]], dtype=np.float32)
            
        #     Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
        #                    [0, 1, 0],
        #                    [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
            
        #     Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
        #                    [np.sin(rz), np.cos(rz), 0],
        #                    [0, 0, 1]], dtype=np.float32)
            
        #     # Combined rotation matrix
        #     R = Rz @ Ry @ Rx
        #     return R

        # # Create rotation matrices for other views
        # rot_0 = rotate_matrix(270, 0, 0)      # Front view
        # rot_90 = rotate_matrix(270, 0, 90)    # Right view
        # rot_180 = rotate_matrix(270, 0, 180)  # Back view
        # rot_270 = rotate_matrix(270, 0, 270)  # Left view


        # train_camera1.R = rot_0
        # train_camera2.R = rot_90 
        # train_camera3.R = rot_180
        # train_camera4.R = rot_270
        
        # train_cameras=[train_camera1,train_camera2,train_camera3,train_camera4]
        train_cameras=scene.getTrainCameras()
        import time
        for i,cam in enumerate(train_cameras):
            
            np.random.seed(int(time.time() * 1000)%10000 + i)  # Set seed based on current time + camera index
            
            # Random rotation angles between 0-10 degrees for each axis
            rx = np.random.uniform(5, 30) * np.pi / 180  # Convert to radians
            ry = np.random.uniform(5, 30) * np.pi / 180
            rz = np.random.uniform(5, 30) * np.pi / 180
            
            # Create rotation matrices
            Rx = torch.tensor([[1, 0, 0],
                     [0, np.cos(rx), -np.sin(rx)],
                     [0, np.sin(rx), np.cos(rx)]], dtype=torch.float32)
            
            Ry = torch.tensor([[np.cos(ry), 0, np.sin(ry)],
                     [0, 1, 0],
                     [-np.sin(ry), 0, np.cos(ry)]], dtype=torch.float32)
            
            Rz = torch.tensor([[np.cos(rz), -np.sin(rz), 0],
                     [np.sin(rz), np.cos(rz), 0],
                     [0, 0, 1]], dtype=torch.float32)
            
            # Combine rotations with existing R matrix
            R_random = Rz @ Ry @ Rx
            train_cameras[i].R = train_cameras[i].R @ R_random.numpy()
            
            # Random translations between 0-0.1
            train_cameras[i].T[0] = np.random.uniform(0.1, 0.3)
            train_cameras[i].T[1] = np.random.uniform(0.1, 0.3) 
            train_cameras[i].T[2] = np.random.uniform(0.1, 0.3)
            train_cameras[i].world_view_transform = torch.tensor(getWorld2View2(train_cameras[i].R, train_cameras[i].T, train_cameras[i].trans, train_cameras[i].scale)).transpose(0, 1).cuda()
            train_cameras[i].full_proj_transform = (train_cameras[i].world_view_transform.unsqueeze(0).bmm(train_cameras[i].projection_matrix.unsqueeze(0))).squeeze(0)
            train_cameras[i].camera_center = train_cameras[i].world_view_transform.inverse()[3, :3]
        # import ipdb
        # ipdb.set_trace()

        
        
        render_set(dataset.model_path, "ablation", scene.loaded_iter, train_cameras,
        None,None,None, gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--frame", type=int, default = 0)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,frame=args.frame)