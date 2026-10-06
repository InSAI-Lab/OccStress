import os
import tempfile
import time
import atexit
import subprocess

import numpy as np
import cv2 as cv
import torch

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    from visualizer.occupancy_visualizer import OccupancyVisualizer
except ImportError:
    OccupancyVisualizer = None

color_map = np.array(
    [
        [0, 0, 0],          # unlabeled            black
        [255, 120, 50],     # barrier              orange
        [255, 192, 203],    # bicycle              pink
        [255, 255, 0],      # bus                  yellow
        [0, 150, 245],      # car                  blue
        [0, 255, 255],      # construction_vehicle cyan
        [255, 127, 0],      # motorcycle           dark orange
        [255, 0, 0],        # pedestrian           red
        [255, 240, 150],    # traffic_cone         light yellow
        [135, 60, 0],       # trailer              brown
        [160, 32, 240],     # truck                purple
        [255, 0, 255],      # driveable_surface    dark pink
        [139, 137, 137],    # other_flat           dark red
        [75, 0, 75],        # sidewalk             dard purple
        [150, 240, 80],     # terrain              light green
        [230, 230, 250],    # manmade              white
        [0, 175, 0],        # vegetation           green
        [255, 255, 255]     # free                 white
    ]
)

occ3d_colors_map = color_map[:, ::-1].copy()
_XVFB_PROCESS = None
_XVFB_DISPLAY = None

def change_occupancy_to_bev(occ_semantics, occ_size=(200, 200, 16), free_cls=17):
    # free_cls == 16 as default

    semantics_valid = np.logical_not(occ_semantics == free_cls)
    d = np.arange(occ_size[-1]).reshape(1, 1, occ_size[-1])
    d = np.repeat(d, occ_size[0], axis=0)
    d = np.repeat(d, occ_size[1], axis=1).astype(np.float32)
    d = d * semantics_valid
    selected = np.argmax(d, axis=2)
    selected_torch = torch.from_numpy(selected)
    semantics_torch = torch.from_numpy(occ_semantics)

    occ_bev_torch = torch.gather(semantics_torch, dim=2, index=selected_torch.unsqueeze(-1))
    occ_bev = occ_bev_torch.numpy()
    occ_bev = occ_bev.flatten().astype(np.int32)
    occ_bev_vis = color_map[occ_bev].astype(np.uint8)
    occ_bev_vis = occ_bev_vis.reshape(occ_size[0], occ_size[1], 3)[::-1, ::-1, :3]
    occ_bev_vis = cv.resize(occ_bev_vis, (occ_size[0], occ_size[1]))
    occ_bev_vis = cv.cvtColor(occ_bev_vis, cv.COLOR_RGB2BGR)

    return occ_bev_vis


def _cleanup_xvfb():
    global _XVFB_PROCESS
    if _XVFB_PROCESS is not None and _XVFB_PROCESS.poll() is None:
        _XVFB_PROCESS.terminate()
        try:
            _XVFB_PROCESS.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _XVFB_PROCESS.kill()
    _XVFB_PROCESS = None


atexit.register(_cleanup_xvfb)


def _ensure_virtual_display():
    global _XVFB_PROCESS, _XVFB_DISPLAY
    if os.environ.get('DISPLAY'):
        return
    if _XVFB_PROCESS is not None and _XVFB_PROCESS.poll() is None and _XVFB_DISPLAY is not None:
        os.environ['DISPLAY'] = _XVFB_DISPLAY
        return

    for display_id in range(90, 120):
        display = f':{display_id}'
        lock_file = f'/tmp/.X{display_id}-lock'
        if os.path.exists(lock_file):
            continue
        try:
            proc = subprocess.Popen(
                ['Xvfb', display, '-screen', '0', '1920x1080x24'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise RuntimeError('Xvfb is required for headless Open3D rendering but was not found.') from exc

        time.sleep(0.5)
        if proc.poll() is None:
            _XVFB_PROCESS = proc
            _XVFB_DISPLAY = display
            os.environ['DISPLAY'] = display
            return

    raise RuntimeError('Failed to start Xvfb for headless Open3D rendering.')


def _darken_color(color, factor):
    color = np.asarray(color, dtype=np.float32) * factor
    return tuple(np.clip(color, 0, 255).astype(np.uint8).tolist())


def change_occupancy_to_oblique_view(occ_semantics,
                                     occ_size=(200, 200, 16),
                                     free_cls=17,
                                     canvas_size=(1100, 780),
                                     bg_color=(250, 250, 250)):
    occ_semantics = np.asarray(occ_semantics)
    canvas_h, canvas_w = canvas_size[1], canvas_size[0]
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

    valid_mask = np.logical_and(occ_semantics != free_cls, occ_semantics != 255)
    coords = np.argwhere(valid_mask)
    if coords.size == 0:
        return canvas

    labels = occ_semantics[valid_mask].astype(np.int32)
    colors = color_map[labels][:, ::-1]

    x = coords[:, 0].astype(np.float32)
    y = coords[:, 1].astype(np.float32)
    z = coords[:, 2].astype(np.float32)

    # Oblique projection with a fixed camera-like direction.
    x_centered = x - occ_semantics.shape[0] / 2.0
    y_centered = y - occ_semantics.shape[1] / 2.0
    proj_x = x_centered - y_centered
    proj_y = (x_centered + y_centered) * 0.58 - z * 3.0

    proj_x -= proj_x.min()
    proj_y = proj_y.max() - proj_y
    margin = 42.0
    scale_x = (canvas_w - 2 * margin) / max(proj_x.max(), 1.0)
    scale_y = (canvas_h - 2 * margin) / max(proj_y.max(), 1.0)
    scale = max(1.0, min(scale_x, scale_y))
    proj_x = proj_x * scale + margin
    proj_y = proj_y * scale + margin

    tile_w = max(3, int(round(scale * 1.45)))
    tile_h = max(2, int(round(scale * 0.82)))
    tile_z = max(4, int(round(scale * 1.55)))

    floor = np.array([
        [margin * 0.9, canvas_h - margin * 0.95],
        [canvas_w * 0.55, canvas_h - margin * 1.55],
        [canvas_w - margin * 0.9, canvas_h - margin * 0.95],
        [canvas_w * 0.45, canvas_h - margin * 0.35],
    ], dtype=np.int32)
    cv.fillConvexPoly(canvas, floor, (242, 242, 242))
    cv.polylines(canvas, [floor], True, (225, 225, 225), 2)

    for grid_ratio in np.linspace(0.12, 0.88, 6):
        start = (
            int(floor[0, 0] * (1 - grid_ratio) + floor[3, 0] * grid_ratio),
            int(floor[0, 1] * (1 - grid_ratio) + floor[3, 1] * grid_ratio),
        )
        end = (
            int(floor[1, 0] * (1 - grid_ratio) + floor[2, 0] * grid_ratio),
            int(floor[1, 1] * (1 - grid_ratio) + floor[2, 1] * grid_ratio),
        )
        cv.line(canvas, start, end, (232, 232, 232), 1)
        start = (
            int(floor[0, 0] * (1 - grid_ratio) + floor[1, 0] * grid_ratio),
            int(floor[0, 1] * (1 - grid_ratio) + floor[1, 1] * grid_ratio),
        )
        end = (
            int(floor[3, 0] * (1 - grid_ratio) + floor[2, 0] * grid_ratio),
            int(floor[3, 1] * (1 - grid_ratio) + floor[2, 1] * grid_ratio),
        )
        cv.line(canvas, start, end, (232, 232, 232), 1)

    depth = x + y + z
    order = np.argsort(depth)
    for idx in order:
        cx = int(round(proj_x[idx]))
        cy = int(round(proj_y[idx]))
        color = tuple(int(v) for v in colors[idx])

        shadow = np.array([
            [cx, cy + tile_h],
            [cx + tile_w, cy + tile_h + tile_h // 2],
            [cx, cy + tile_h * 2],
            [cx - tile_w, cy + tile_h + tile_h // 2],
        ], dtype=np.int32)

        top = np.array([
            [cx, cy - tile_z],
            [cx + tile_w, cy - tile_z // 2],
            [cx, cy],
            [cx - tile_w, cy - tile_z // 2],
        ], dtype=np.int32)
        right = np.array([
            [cx + tile_w, cy - tile_z // 2],
            [cx + tile_w, cy - tile_z // 2 + tile_h],
            [cx, cy + tile_h],
            [cx, cy],
        ], dtype=np.int32)
        left = np.array([
            [cx - tile_w, cy - tile_z // 2],
            [cx, cy],
            [cx, cy + tile_h],
            [cx - tile_w, cy - tile_z // 2 + tile_h],
        ], dtype=np.int32)

        cv.fillConvexPoly(canvas, shadow, (228, 228, 228))
        cv.fillConvexPoly(canvas, left, _darken_color(color, 0.72))
        cv.fillConvexPoly(canvas, right, _darken_color(color, 0.86))
        cv.fillConvexPoly(canvas, top, color)
        cv.polylines(canvas, [left], True, _darken_color(color, 0.52), 1)
        cv.polylines(canvas, [right], True, _darken_color(color, 0.62), 1)
        cv.polylines(canvas, [top], True, _darken_color(color, 0.74), 1)

    return canvas


def render_occupancy_to_3d(occ_semantics,
                           free_cls=17,
                           save_dir=None,
                           view_json_path='view.json',
                           voxel_size=(0.4, 0.4, 0.4),
                           occ_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
                           background_color=(255, 255, 255)):
    if o3d is None or OccupancyVisualizer is None:
        raise ImportError('open3d visualization dependencies are not installed.')
    if save_dir is None:
        save_dir = tempfile.gettempdir()
    os.makedirs(save_dir, exist_ok=True)
    _ensure_virtual_display()

    param = o3d.io.read_pinhole_camera_parameters(view_json_path) if os.path.exists(view_json_path) else None
    occ_visualizer = OccupancyVisualizer(
        color_map=occ3d_colors_map,
        background_color=background_color,
    )

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=save_dir) as f:
        tmp_path = f.name

    try:
        occ_visualizer.vis_occ(
            occ_semantics,
            occ_flow=None,
            ignore_labels=[free_cls, 255],
            voxelSize=voxel_size,
            range=list(occ_range),
            save_path=tmp_path,
            wait_time=1,
            view_json=param,
            car_model_mesh=None,
        )
        param = occ_visualizer.o3d_vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(view_json_path, param)
        occ_visualizer.o3d_vis.destroy_window()
        image = cv.imread(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if image is None:
        raise RuntimeError(f'Failed to render occupancy visualization to {tmp_path}')
    return image
