import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, SMPL_MEAN_PARAMS, H36M_TO_J14
from lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat

class SMPLRegressor(nn.Module):
    def __init__(self, dim_rep=256, num_joints=17, hidden_dim=2048, dropout_ratio=0.):
        super(SMPLRegressor, self).__init__()
        param_pose_dim = 24 * 6

        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(num_joints*dim_rep, hidden_dim)
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.fc2 = nn.Linear(num_joints*dim_rep, hidden_dim)
        self.fc3 = nn.Linear(dim_rep, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.bn3 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.head_pose = nn.Linear(hidden_dim, param_pose_dim)
        self.head_shape = nn.Linear(hidden_dim, 10)
        self.head_cam = nn.Linear(dim_rep, 3)
        nn.init.xavier_uniform_(self.head_pose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.head_shape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.head_cam.weight, gain=0.01)
        
        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False,
        )
        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, joint_feat, img_feat, is_train=False, J_regressor=None):
        B, T, J, C = joint_feat.shape
        BT = B * T
        joint_feat = joint_feat.reshape(B, T, -1)

        feat_pose = joint_feat.reshape(BT, -1)     # [BT, JC]

        feat_pose = self.dropout(feat_pose)
        feat_pose = self.fc1(feat_pose)
        feat_pose = self.bn1(feat_pose)
        feat_pose = self.relu1(feat_pose)           # [BT, JC]

        feat_shape = joint_feat.permute(0,2,1)      # [B, T, JC] -> [B, JC, T]
        feat_shape = self.pool2(feat_shape).reshape(B, -1)          # [B, JC]

        feat_shape = self.dropout(feat_shape)
        feat_shape = self.fc2(feat_shape)
        feat_shape = self.bn2(feat_shape)
        feat_shape = self.relu2(feat_shape)

        pred_pose = self.init_pose.expand(BT, -1)   # (NT, C)
        pred_shape = self.init_shape.expand(B, -1)  # (N, C)
        pred_cam = self.init_cam.expand(BT, -1)

        pred_pose = self.head_pose(feat_pose) + pred_pose
        pred_shape = self.head_shape(feat_shape) + pred_shape
        pred_shape = pred_shape.expand(T, B, -1).permute(1, 0, 2).reshape(BT, -1)

        # Cam
        img_feat = self.dropout(img_feat)
        img_feat = self.fc3(img_feat)
        img_feat = self.bn3(img_feat)
        img_feat = self.relu3(img_feat)
        pred_cam = self.head_cam(img_feat) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(-1, 24, 3, 3)
        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        if not is_train and J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
        pred_keypoints_2d = projection(pred_joints, pred_cam)

        output = [{
            'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1),  # [BT, 85]
            'verts'  : pred_vertices,                                   # [BT, 6890, 3]
            'kp_2d'  : pred_keypoints_2d,                               # [BT, 49, 2]
            'kp_3d'  : pred_joints,                                     # [BT, 49, 3]
            'rotmat' : pred_rotmat                                      # [BT, 24, 3, 3]
        }]
        return output
    
def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]