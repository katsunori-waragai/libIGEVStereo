import numpy as np
import cv2
import open3d as o3d

def reproject_point_cloud(point_cloud, color, right_camera_intrinsics, baseline):
    """
    点群データを右カメラ視点に再投影する関数

    Args:
        point_cloud: 点群データ (Nx3 numpy array)
        color: 点の色情報 (Nx3 numpy array)
        right_camera_intrinsics: 右カメラの内部パラメータ
        baseline: 基線長

    Returns:
        reprojected_image: 再投影画像
    """

    # ... (これまでのコードと同様)

    # 視差から深度を計算
    depth = baseline * right_camera_intrinsics[0, 0] / points_2d[:, 0]

    # 深度とカメラ座標系との関係から、3D座標を再計算
    # (ステレオ平行化済みなので、Z座標は深度と一致)
    reprojected_point_cloud = np.hstack([points_2d, depth.reshape(-1, 1)])

    # ... (残りのコードはほぼ同様)

# 使用例
# ... (これまでのコードと同様)

# 基線長の設定
baseline = 0.1  # カメラ間の距離

# 再投影
reprojected_image = reproject_point_cloud(point_cloud, color, right_camera_intrinsics, baseline)
