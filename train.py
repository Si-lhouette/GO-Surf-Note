import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from model.sdf_grid_model import SDFGridModel, qp_to_sdf
from config import load_config
from model.utils import matrix_to_pose6d, pose6d_to_matrix
from model.utils import coordinates
from dataio.scannet_dataset import ScannetDataset
from dataio.rgbd_dataset import RGBDDataset

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
import math
import numpy as np

def main(args):
    # scene 确定数据集，exp_name 确定一次train
    config = load_config(scene=args.scene, exp_name=args.exp_name)
        
    # 注册ROS节点，实时可视化消息
    rospy.init_node('train_node', anonymous=True)
    all_map_pub = rospy.Publisher('/sdf', PointCloud2, queue_size=1)

    # 初始化tensorboard 的 SummaryWriter 用于训练过程可视化
    # 训练中间过程数据将会被记录在 events_save_dir 下，运行“tensorboard --logdir=日志目录路径”可以得到可视化界面
    events_save_dir = os.path.join(config["log_dir"], "events")
    if not os.path.exists(events_save_dir):
        os.makedirs(events_save_dir)
    writer = SummaryWriter(log_dir=events_save_dir)
    
    # 设置训练设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 前缀 f 是用于格式化字符串字面值的标识符。在一个以 f 开头的字符串中，可以在字符串中使用花括号 {} 来插入变量或表达式的值
    print(f"Using device: {device}")

    # 载入数据集
    # 支持scannet数据集 or 普通rgbd数据集
    if config["dataset_type"] == "scannet":
        dataset = ScannetDataset(os.path.join(config["datasets_dir"], args.scene), trainskip=config["trainskip"], device=torch.device("cpu"))
    elif config["dataset_type"] == "rgbd":
        dataset = RGBDDataset(os.path.join(config["datasets_dir"], args.scene), trainskip=config["trainskip"],
            downsample_factor=config["downsample_factor"], device=torch.device("cpu"))
    else:
        raise NotImplementedError
    
    # 载入网络模型
    model = SDFGridModel(config, device, dataset.get_bounds())
    
    # 将数据以每条ray为单位打乱
    ray_indices = torch.randperm(len(dataset) * dataset.H * dataset.W) # randperm will random the input data
    
    # 设置优化器
    # Inverse sigma from NeuS paper
    # 在这里定义了三个不同的参数组，每个参数组都有不同的学习率（lr），之后在每一次step()时，三个部分顺序进行串行优化
    # inv_s 和后面计算loss的时候的权重相关
    inv_s = nn.parameter.Parameter(torch.tensor(0.3, device=device))
    optimizer = torch.optim.Adam([{"params": model.decoder.parameters(), "lr": config["lr"]["decoder"]},
                                  {"params": model.grid.parameters(), "lr": config["lr"]["features"]},
                                  {"params": inv_s, "lr": config["lr"]["inv_s"]}])
    
    # 与优化poses 相关部分
    optimise_poses = config["optimise_poses"]
    poses_mat_init = torch.stack(dataset.c2w_list, dim=0).to(device)
    
    if optimise_poses:
        poses = nn.Parameter(matrix_to_pose6d(poses_mat_init))
        poses_optimizer = torch.optim.Adam([poses], config["lr"]["poses"])

    if args.start_iter > 0:
        state = torch.load(os.path.join(config["checkpoints_dir"], "chkpt_{}".format(args.start_iter)), map_location=device)
        inv_s = state["inv_s"]
        model.load_state_dict(state["model"])
        iteration = state["iteration"]
        optimizer.load_state_dict(state["optimizer"])
        
        if optimise_poses:
            poses = state["poses"]
            poses_optimizer.load_state_dict(state["poses_optimizer"])
    else:
        print("Pre-Train SDF of a sphere")
        center = model.world_dims / 2. + model.volume_origin
        radius = model.world_dims.min() / 2.
        
        # Train SDF of a sphere 用一个球壳来初始化SDF
        for _ in range(500):
            optimizer.zero_grad()
            # 返回体素网格中的所有体素的三维坐标（dim_x * dim_y * dim_z个）
            coords = coordinates(model.voxel_dims[1] - 1, device).float().t()
            # torch.rand_like() 函数用于创建一个和输入张量具有相同形状的张量，其中的值是在 [0, 1) 范围内均匀随机分布
            pts = (coords + torch.rand_like(coords)) * config["voxel_sizes"][1] + model.volume_origin
            sdf, *_ = qp_to_sdf(pts.unsqueeze(1), model.volume_origin, model.world_dims, model.grid, model.sdf_decoder,
                                concat_qp=config["decoder"]["geometry"]["concat_qp"], rgb_feature_dim=config["rgb_feature_dim"])
            sdf = sdf.squeeze(-1) # squeeze will get rid of the invalid dim(matrix: 2*1*1 -> 2*1)
            # 计算真值
            target_sdf = radius - (center - pts).norm(dim=-1)
            loss = torch.nn.functional.mse_loss(sdf, target_sdf)
            
            if loss.item() < 1e-10:
                break
            
            loss.backward()
            optimizer.step()
        
        print("Init loss after geom init (sphere)", loss.item())
    
        # Reset optimizer
        optimizer = torch.optim.Adam([{"params": model.decoder.parameters(), "lr": config["lr"]["decoder"]},
                                      {"params": model.grid.parameters(), "lr": config["lr"]["features"]},
                                      {"params": inv_s, "lr": config["lr"]["inv_s"]}])
        
    img_stride = dataset.H * dataset.W
    # ray_indices.shape[0] is the number of all the ray of all the frame 
    n_batches = ray_indices.shape[0] // config["batch_size"] # 双斜杠是整数除法操作符，它执行除法并返回结果的整数部分，而忽略小数部分
    # The iteratioin is for the set of all the ray
    for iteration in trange(args.start_iter + 1, config["iterations"] + 1):
        batch_idx = iteration % n_batches
        print("batch_idx: ", batch_idx, "(" , n_batches, "), idx_range: [",
         (batch_idx * config["batch_size"]), "->", ((batch_idx + 1) * config["batch_size"]))
        ray_ids = ray_indices[(batch_idx * config["batch_size"]):((batch_idx + 1) * config["batch_size"])] 
        frame_id = ray_ids.div(img_stride, rounding_mode='floor')
        print("frame_size: ", frame_id.size())
        # 读取像素坐标
        v = (ray_ids % img_stride).div(dataset.W, rounding_mode='floor')
        u = ray_ids % img_stride % dataset.W
        
        depth = dataset.depth_list[frame_id, v, u].to(device, non_blocking=True)
        rgb = dataset.rgb_list[frame_id, :, v, u].to(device, non_blocking=True)
        
        # 读取内参
        fx, fy = dataset.K_list[frame_id, 0, 0], dataset.K_list[frame_id, 1, 1]
        cx, cy = dataset.K_list[frame_id, 0, 2], dataset.K_list[frame_id, 1, 2]

        if config["dataset_type"] == "scannet":  # OpenCV
            rays_d_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(fx)], dim=-1).to(device)
        else:  # OpenGL 将图像平面坐标系转换到相机坐标系下
            rays_d_cam = torch.stack([(u - cx) / fx, -(v - cy) / fy, -torch.ones_like(fy)], dim=-1).to(device)
        
        if optimise_poses:
            batch_poses = poses[frame_id]
            c2w = pose6d_to_matrix(batch_poses)
        else:
            c2w = poses_mat_init[frame_id]
        
        # ray 起点
        rays_o = c2w[:,:3,3] # 这里写成c2w[0:3,3]不是一样的吗，会提取出转移矩阵的位置部分
        # ray 终点
        rays_d = torch.bmm(c2w[:, :3, :3], rays_d_cam[..., None]).squeeze()
        
        # 调用model的forward函数
        ret = model(rays_o, rays_d, rgb, depth, inv_s=torch.exp(10. * inv_s),
                    smoothness_std=config["smoothness_std"], iter=iteration)

        loss = config["rgb_weight"] * ret["rgb_loss"] +\
               config["depth_weight"] * ret["depth_loss"] +\
               config["fs_weight"] * ret["fs_loss"] +\
               config["sdf_weight"] * ret["sdf_loss"] +\
               config["normal_regularisation_weight"] * ret["normal_regularisation_loss"] +\
               config["normal_supervision_weight"] * ret["normal_supervision_loss"] +\
               config["eikonal_weight"] * ret["eikonal_loss"]
        
        loss.backward()

        # 梯度裁剪：梯度的范数不超过 1.0
        torch.nn.utils.clip_grad_norm_(model.grid.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.)
        
        optimizer.step()
        optimizer.zero_grad()
        
        if optimise_poses:
            if iteration > 100:
                if iteration % 3 == 0:
                    poses_optimizer.step()
                    poses_optimizer.zero_grad()
            else:
                poses_optimizer.zero_grad()

        writer.add_scalar('depth', ret["depth_loss"].item(), iteration)
        writer.add_scalar('rgb', ret["rgb_loss"].item(), iteration)
        writer.add_scalar('fs', ret["fs_loss"].item(), iteration)
        writer.add_scalar('sdf', ret["sdf_loss"].item(), iteration)
        writer.add_scalar('psnr', ret["psnr"].item(), iteration)
        writer.add_scalar('eikonal', ret["eikonal_loss"].item(), iteration)
        writer.add_scalar('normal regularisation', ret["normal_regularisation_loss"].item(), iteration)

        if iteration % args.i_print == 0:
            tqdm.write("Iter: {}, PSNR: {:6f}, RGB Loss: {:6f}, Depth Loss: {:6f}, SDF Loss: {:6f}, FS Loss: {:6f}, "
                       "Eikonal Loss: {:6f}, Smoothness Loss: {:6f}".format(iteration,
                                                                            ret["psnr"].item(),
                                                                            ret["rgb_loss"].item(),
                                                                            ret["depth_loss"].item(),
                                                                            ret["sdf_loss"].item(),
                                                                            ret["fs_loss"].item(),
                                                                            ret["eikonal_loss"].item(),
                                                                            ret["normal_regularisation_loss"].item()))
            
            # visualize SDF
            print("Visual SDF")
            center = model.world_dims / 2. + model.volume_origin
            radius = model.world_dims.min() / 2.
            print("center: ", center, ", radius: ", radius, ", origin: ", model.volume_origin)

            # for _ in range(500):
            dims = model.voxel_dims[1].clone()
            print("dim: ", dims)
            dims[2] = 2
            print("dim: ", dims)
            print("origin dim: ", model.voxel_dims[1])

            coords = coordinates(dims - 1, device).float().t()
            print("coor: ", coords)

            points = []
            for h in [-3.0, -2.0, -1.0, 0.0]:
                origin = model.volume_origin.clone()
                origin[2] = h

                pts = (coords) * config["voxel_sizes"][1] + origin
                print("pts: ", pts)
                sdf, *_ = qp_to_sdf(pts.unsqueeze(1), model.volume_origin, model.world_dims, model.grid, model.sdf_decoder,
                                    concat_qp=config["decoder"]["geometry"]["concat_qp"], rgb_feature_dim=config["rgb_feature_dim"])
                sdf = sdf.squeeze(-1) # squeeze will get rid of the invalid dim(matrix: 2*1*1 -> 2*1)
                print("sdf: ", sdf)
                sdf_np = sdf.cpu().detach().numpy()
                origin[1] += h * 4.0
                pts = pts + origin
                pts_np = pts.cpu().detach().numpy()

                for i in range(1,len(sdf)) :
                    points.append([pts_np[i,0], pts_np[i,1], pts_np[i,2], sdf_np[i]])

            # transfer to pcl
            map = PointCloud2()
            map.height = 1
            map.width = len(points)
            map.fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField("intensity", 12, PointField.FLOAT32, 1)]
            map.point_step = 16  # 12
            map.row_step = 16 * len(points)
            map.is_bigendian = False
            map.is_dense = False
            map.data = np.asarray(points, np.float32).tostring()
            map.header.frame_id = "world"
            all_map_pub.publish(map)


        # Save checkpoint
        if iteration % args.i_save == 0:
            state = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'iteration': iteration,
                     'poses': poses if 'poses' in locals() else None,
                     'poses_optimizer': poses_optimizer.state_dict() if 'poses_optimizer' in locals() else None,
                     'inv_s': inv_s}
            torch.save(state, os.path.join(config["checkpoints_dir"], "chkpt_{}".format(iteration)))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="release_test")
    parser.add_argument('--scene', type=str, default="grey_white_room")
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--i_print', type=int, default=20)
    parser.add_argument('--i_save', type=int, default=1000)
    args = parser.parse_args()
    try:
        main(args)
    except rospy.ROSInterruptException:
        pass