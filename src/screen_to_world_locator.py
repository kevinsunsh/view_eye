import numpy as np

def deproject_screen_to_world(
    screen_pos: np.ndarray,           # [x, y] 像素坐标
    view_rect: tuple,                 # (min_x, min_y, width, height)
    inv_view_matrix: np.ndarray,      # 4x4 逆视图矩阵 (world = inv_view * camera)
    inv_projection_matrix: np.ndarray # 4x4 逆投影矩阵 (camera = inv_proj * proj)
) -> tuple[np.ndarray, np.ndarray]:
    """
    将屏幕坐标反向投影到世界空间中的射线起点和方向。

    Args:
        screen_pos: [x, y] 屏幕像素坐标
        view_rect: (min_x, min_y, width, height) 视口区域
        inv_view_matrix: 4x4 逆视图矩阵 (将相机空间转为世界空间)
        inv_projection_matrix: 4x4 逆投影矩阵 (将投影空间转为相机空间)

    Returns:
        world_origin: 世界空间中射线起点（即相机位置）
        world_direction: 世界空间中射线方向（单位向量）
    """
    # 1. 转换为整数像素坐标
    pixel_x = int(screen_pos[0])
    pixel_y = int(screen_pos[1])

    # 2. 归一化到 [0, 1] 范围内
    min_x, min_y, width, height = view_rect
    normalized_x = (pixel_x - min_x) / width
    normalized_y = (pixel_y - min_y) / height

    # 3. 映射到 [-1, 1] 的投影空间（NDC）
    screen_space_x = (normalized_x - 0.5) * 2.0
    screen_space_y = (1.0 - normalized_y - 0.5) * 2.0  # 注意 Y 轴翻转

    # 4. 构造投影空间中的两个点（z=1 和 z=0.01）
    ray_start_proj = np.array([screen_space_x, screen_space_y, 1.0, 1.0])  # near plane
    ray_end_proj   = np.array([screen_space_x, screen_space_y, 0.01, 1.0])  # far point

    # 5. 应用逆投影矩阵（投影空间 → 相机空间）
    h_ray_start_view = inv_projection_matrix @ ray_start_proj
    h_ray_end_view   = inv_projection_matrix @ ray_end_proj

    # 6. 除以 w 分量，得到相机空间坐标
    if h_ray_start_view[3] != 0:
        ray_start_view = h_ray_start_view[:3] / h_ray_start_view[3]
    else:
        ray_start_view = h_ray_start_view[:3]

    if h_ray_end_view[3] != 0:
        ray_end_view = h_ray_end_view[:3] / h_ray_end_view[3]
    else:
        ray_end_view = h_ray_end_view[:3]

    # 7. 计算相机空间中的方向
    ray_dir_view = ray_end_view - ray_start_view
    ray_dir_view = ray_dir_view / np.linalg.norm(ray_dir_view)

    # 8. 变换到世界空间
    ray_start_world = inv_view_matrix @ np.append(ray_start_view, 1.0)
    ray_start_world = ray_start_world[:3] / ray_start_world[3] if ray_start_world[3] != 0 else ray_start_world[:3]

    ray_dir_world = inv_view_matrix @ np.append(ray_dir_view, 0.0)
    ray_dir_world = ray_dir_world[:3] / np.linalg.norm(ray_dir_world)

    return ray_start_world, ray_dir_world


def world_position_from_depth(
    screen_pos: np.ndarray,
    view_rect: tuple,
    inv_view_matrix: np.ndarray,
    inv_projection_matrix: np.ndarray,
    camera_position: np.ndarray,
    camera_forward: np.ndarray,
    depth_value: float,
):
    """
    基于射线 + 深度余弦校正计算最终世界坐标。

    约定：depth_value 为相机前向方向（camera_forward）上的线性深度(米)，
    即常见的相机空间 Z 深度。如果是其它定义，请先转换到该定义。
    """
    origin, direction = deproject_screen_to_world(
        screen_pos, view_rect, inv_view_matrix, inv_projection_matrix
    )
    # 使用更准确的相机位置作为射线起点
    cam_pos = np.asarray(camera_position, dtype=np.float64)
    cam_fwd = np.asarray(camera_forward, dtype=np.float64)
    cam_fwd = cam_fwd / (np.linalg.norm(cam_fwd) if np.linalg.norm(cam_fwd) > 0 else 1.0)

    correction_factor = float(np.dot(direction, cam_fwd))
    if abs(correction_factor) < 1e-6:
        raise ValueError("Ray direction is nearly perpendicular to camera forward vector.")

    euclidean_distance = float(depth_value) / correction_factor
    world_position = cam_pos + direction * euclidean_distance
    return world_position

def screen_distance_to_world_distance(
    screen_pos1: np.ndarray,
    screen_pos2: np.ndarray,
    depth1: float,
    depth2: float,
    view_rect: tuple,
    inv_view_matrix: np.ndarray,
    inv_projection_matrix: np.ndarray,
    camera_position: np.ndarray,
    camera_forward: np.ndarray,
) -> float:
    """
    根据屏幕空间的两个点和对应的深度值，计算它们在世界空间中的距离。
    
    Args:
        screen_pos1: 第一个屏幕点坐标 [x, y]
        screen_pos2: 第二个屏幕点坐标 [x, y]
        depth1: 第一个点的深度值（米）
        depth2: 第二个点的深度值（米）
        view_rect: 视口区域 (min_x, min_y, width, height)
        inv_view_matrix: 4x4 逆视图矩阵
        inv_projection_matrix: 4x4 逆投影矩阵
        camera_position: 相机位置
        camera_forward: 相机前向向量
        
    Returns:
        两个点在世界空间中的欧几里得距离（米）
    """
    # 将两个屏幕点转换为世界坐标
    world_pos1 = world_position_from_depth(
        screen_pos1, view_rect, inv_view_matrix, inv_projection_matrix,
        camera_position, camera_forward, depth1
    )
    
    world_pos2 = world_position_from_depth(
        screen_pos2, view_rect, inv_view_matrix, inv_projection_matrix,
        camera_position, camera_forward, depth2
    )
    
    # 计算欧几里得距离
    distance = np.linalg.norm(world_pos2 - world_pos1)
    return float(distance)


class _Compat:
    @staticmethod
    def pixel_depth_to_world_with_forward(
        screen_pos: np.ndarray,
        depth_value: float,
        view_rect: tuple,
        inv_view_matrix: np.ndarray,
        inv_projection_matrix: np.ndarray,
        camera_position: np.ndarray,
        camera_forward: np.ndarray,
    ) -> np.ndarray:
        return world_position_from_depth(
            screen_pos,
            view_rect,
            inv_view_matrix,
            inv_projection_matrix,
            camera_position,
            camera_forward,
            depth_value,
        )
