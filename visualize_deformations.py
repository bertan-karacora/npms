from pathlib import Path
import numpy as np
import open3d as o3d
import torch

from npms.body_model.smpl.smpl import Smpl

# Setup
filename_tris = "smpl_tris.npy"

dir_data = "data"
dir_set = "cape_release"
dir_sequences = "sequences"
dir_identity = "00032"
dir_sequence = "shortshort_chicken_wings"
dir_misc = "misc"

path_set = Path(".") / dir_data / dir_set
path_sequence = path_set / dir_sequences / dir_identity / dir_sequence
path_tris = path_set / dir_misc / filename_tris

frames = list(path_sequence.glob("*.npz"))
cur = 1

# Run
data = np.load(frames[cur])
points_cano = data["v_cano"]
points_posed = data["v_posed"] - data["transl"]
orient = data["pose"][:3]
pose = data["pose"][3:]
pose_cano = np.zeros_like(pose)
pose_cano[2], pose_cano[5] = 0.5, -0.5

faces = np.load(path_tris)

body_model = Smpl(device="cuda:0")
points_cano_torch = torch.from_numpy(points_cano).cuda().type(torch.float32)
points_posed_torch = torch.from_numpy(points_posed).cuda().type(torch.float32)
orient_torch = torch.from_numpy(orient).cuda().type(torch.float32).unsqueeze(0)
pose_cano_torch = torch.from_numpy(pose_cano).cuda().type(torch.float32).unsqueeze(0)
pose_torch = torch.from_numpy(pose).cuda().type(torch.float32).unsqueeze(0)
vertices_cano = body_model(body_pose=pose_cano_torch, t_pose=points_cano_torch).cpu().numpy()
vertices_posed = body_model(orient=orient_torch, body_pose=pose_torch, t_pose=points_cano_torch).cpu().numpy()

pointcloud_cano = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices_cano))
pointcloud_cano.paint_uniform_color([0, 0, 0])
mesh_cano = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices_cano), o3d.utility.Vector3iVector(faces))
mesh_cano.compute_vertex_normals()

pointcloud_posed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices_posed))
pointcloud_posed.paint_uniform_color([0, 0, 0])
mesh_posed = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices_posed), o3d.utility.Vector3iVector(faces))
mesh_posed.compute_vertex_normals()

num_vertices = points_cano.shape[0]
corresp = [(k, k) for k in range(0, num_vertices)]
deformations = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pointcloud_posed, pointcloud_cano, corresp)
colors = np.zeros((num_vertices, 3))
for i, line in enumerate(np.asarray(deformations.lines)):
    vec = points_posed[line[1] - num_vertices] - points_cano[line[0]]
    vec = vec / np.linalg.norm(vec)
    colors[i] = np.array([np.abs(vec[1]), np.abs(vec[0]), np.abs(vec[2])])
deformations.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([mesh_cano, pointcloud_cano, deformations, mesh_posed, pointcloud_posed])
