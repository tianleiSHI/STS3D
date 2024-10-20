import cv2
import numpy as np
import torch
from PIL import Image
import time
from skimage import measure
import trimesh

COLOR_MAPS_DICT = {
    'Paired': [
        [166, 206, 227], [31, 120, 180], [178, 223, 138], [51, 160, 44], [251, 154, 153],
        [227, 26, 28], [253, 191, 111], [255, 127, 0], [202, 178, 214], [106, 61, 154],
        [255, 255, 153], [177, 89, 40],
    ],
    'Pastel1': [
        [251, 180, 174], [179, 205, 227], [204, 235, 197], [222, 203, 228], [254, 217, 166],
        [255, 255, 204], [229, 216, 189], [253, 218, 236], [242, 242, 242],
    ],
    'Pastel2': [
        [179, 226, 205], [253, 205, 172], [203, 213, 232], [244, 202, 228], [230, 245, 201],
        [255, 242, 174], [241, 226, 204], [204, 204, 204],
    ],
    'Accent': [
        [127, 201, 127], [190, 174, 212], [253, 192, 134], [255, 255, 153], [56, 108, 176],
        [240, 2, 127], [191, 91, 22], [102, 102, 102],
    ],
    'Dark2': [
        [27, 158, 119], [217, 95, 2], [117, 112, 179], [231, 41, 138], [102, 166, 30],
        [230, 171, 2], [166, 118, 29], [102, 102, 102],
    ],
    'Set1': [
        [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163], [255, 127, 0],
        [255, 255, 51], [166, 86, 40], [247, 129, 191], [153, 153, 153],
    ],
    'Set2': [
        [102, 194, 165], [252, 141, 98], [141, 160, 203], [231, 138, 195], [166, 216, 84],
        [255, 217, 47], [229, 196, 148], [179, 179, 179],
    ],
    'Set3': [
        [141, 211, 199], [255, 255, 179], [190, 186, 218], [251, 128, 114], [128, 177, 211],
        [253, 180, 98], [179, 222, 105], [252, 205, 229], [217, 217, 217], [188, 128, 189],
        [204, 235, 197], [255, 237, 111],
    ],
    'tab10': [
        [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189],
        [140, 86, 75], [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207],
    ],
    'tab20': [
        [31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120], [44, 160, 44],
        [152, 223, 138], [214, 39, 40], [255, 152, 150], [148, 103, 189], [197, 176, 213],
        [140, 86, 75], [196, 156, 148], [227, 119, 194], [247, 182, 210], [127, 127, 127],
        [199, 199, 199], [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229],
    ],
    'tab20b': [
        [57, 59, 121], [82, 84, 163], [107, 110, 207], [156, 158, 222], [99, 121, 57],
        [140, 162, 82], [181, 207, 107], [206, 219, 156], [140, 109, 49], [189, 158, 57],
        [231, 186, 82], [231, 203, 148], [132, 60, 57], [173, 73, 74], [214, 97, 107],
        [231, 150, 156], [123, 65, 115], [165, 81, 148], [206, 109, 189], [222, 158, 214],
    ],
    'tab20c': [
        [49, 130, 189], [107, 174, 214], [158, 202, 225], [198, 219, 239], [230, 85, 13],
        [253, 141, 60], [253, 174, 107], [253, 208, 162], [49, 163, 84], [116, 196, 118],
        [161, 217, 155], [199, 233, 192], [117, 107, 177], [158, 154, 200], [188, 189, 220],
        [218, 218, 235], [99, 99, 99], [150, 150, 150], [189, 189, 189], [217, 217, 217],
    ],
}
PALETTE = np.concatenate(list(COLOR_MAPS_DICT.values()), axis=0)



def loose_bbox(coords, image_size, loose_coef=0.):
    w, h = image_size
    coords = np.array(coords)
    roi_w, roi_h = coords[2] - coords[0], coords[3] - coords[1]

    if isinstance(loose_coef, float):
        left, top, right, bottom = loose_coef, loose_coef, loose_coef, loose_coef
    else:
        left, top, right, bottom = loose_coef

    coords[0] -= roi_w * left
    coords[1] -= roi_h * top
    coords[2] += roi_w * right
    coords[3] += roi_h * bottom

    coords[0] = max(0, int(coords[0]))
    coords[1] = max(0, int(coords[1]))
    coords[2] = min(w, int(coords[2]))
    coords[3] = min(h, int(coords[3]))
    coords = coords.astype(np.int)
    return coords


def loose_3d_bbox(coords, size, loose_coef):
    if len(coords.shape) == 3:
        xx, yy, zz = np.where(coords > 0)
        x1, x2 = xx.min(), xx.max()
        y1, y2 = yy.min(), yy.max()
        z1, z2 = zz.min(), zz.max()
        coords = [x1, y1, z1, x2, y2, z2]


    coords = np.array(coords, dtype=int)
    bbox_size = coords[3:] - coords[:3]

    if isinstance(loose_coef, float):
        loose_coef = np.tile(loose_coef, 6)

    for i in range(3):
        coords[i] -= bbox_size[i] * loose_coef[i]
        coords[i] = max(0, int(coords[i]))

    for i in range(3, 6):
        coords[i] += bbox_size[i - 3] * loose_coef[i]
        coords[i] = min(size[i - 3], int(coords[i]))

    return coords


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


# 验证集指标计算
class Metric():
    def __init__(self, num):
        self.num_joints = num
        self.reset()

    def reset(self):
        self.count = np.zeros((self.num_joints,))
        self.dist = np.zeros((self.num_joints,))
        # self.dist = []

    def update(self, pred, gt, weight):
        assert self.num_joints == weight.shape[1]

        weight = (weight != 0).astype(np.float32)
        weight = weight.squeeze()
        self.count += np.sum(weight, axis=0)

        diff = (pred - gt)
        dist = np.linalg.norm(diff, axis=-1)
        dist *= weight
        self.dist += np.sum(dist, axis=0)

    def compute(self):
        avg_dist = self.dist / self.count
        return avg_dist


class AccuracyMetric():
    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.count = np.zeros((self.num_classes,))
        self.acc = np.zeros((self.num_classes,))
        # self.dist = []

    def update(self, pred, gt, weight=None):
        n, c = pred.shape[:2]

        pred = (pred > 0.5)

        assert c == self.num_classes

        pred = pred.reshape(n, c, -1)
        gt = gt.reshape(n, c, -1)
        acc = np.sum(pred == gt, axis=(0, 2))

        self.acc += acc
        self.count += n

    def compute(self):
        avg_dist = self.acc / self.count
        return avg_dist


def vol_to_mesh(vol, spacing=(1., 1., 1.), component_number=-1, step_size=1):
    a, b, c = np.where(vol > 0)
    if len(a) == 0 or len(b) == 0 or len(c) == 0:
        return trimesh.Trimesh()

    a1, a2 = a.min(), a.max() + 1
    b1, b2 = b.min(), b.max() + 1
    c1, c2 = c.min(), c.max() + 1
    vol = vol[a1:a2, b1:b2, c1:c2]

    pad_width = 5
    vol = np.pad(vol, pad_width=pad_width)
    convert_start = time.time()
    verts, faces, normals, values = measure.marching_cubes(
        vol,
        gradient_direction='ascent',
        # spacing=(0.2, 0.2, 0.2),
        spacing=spacing,
        # allow_degenerate=False,
        step_size=step_size
    )
    verts += (np.array([a1, b1, c1]) - pad_width) * spacing
    # print(f'vertices num: {verts.shape[0]}, faces num: {faces.shape[0]}')
    # print(f'convert time {time.time() - convert_start:.2f}s')
    #
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh


def affine_and_clip_bbox(bboxes, shape, affine=None, min_size=16):
    device = bboxes.device
    bboxes = bboxes.cpu().numpy()
    affine = affine.cpu().numpy()
    bboxes = np.sort(bboxes, axis=1)
    if affine is not None:
        n = len(bboxes)
        bboxes = bboxes.reshape(n, -1)
        xyz = bboxes[:, [0, 1, 2, 3, 4, 5,  # x1y1z1, x2y2z2
                         0, 4, 2, 3, 1, 5,  # x1y2z1, x2y1z1
                         0, 1, 5, 3, 4, 2,
                         0, 4, 5, 3, 1, 2, ]
              ].reshape(-1, 3)

        xyz = trimesh.transformations.transform_points(xyz, np.linalg.inv(affine))
        xyz = xyz.reshape((-1, 8, 3))
        bboxes = np.concatenate([xyz.min(axis=1, keepdims=True), xyz.max(axis=1, keepdims=True)], axis=1)
        # bboxes = xyz.reshape((-1, 2, 3))

    width, height, depth = shape
    bboxes[:, :, 0] = bboxes[:, :, 0].clip(0, width)
    bboxes[:, :, 1] = bboxes[:, :, 1].clip(0, height)
    bboxes[:, :, 2] = bboxes[:, :, 2].clip(0, depth)

    whd = bboxes[:, 1] - bboxes[:, 0]
    w, h, d = np.hsplit(whd, (1, 2))
    valid_index = (w > min_size) & (h > min_size) & (d > min_size)
    valid_index = valid_index[:, 0]
    bboxes = bboxes[valid_index]

    # bboxes =
    bboxes = torch.tensor(bboxes).to(device)
    valid_index = torch.tensor(valid_index).to(device)
    return bboxes, valid_index


def extract_bbox_from_mask(mask, min_area=1000):
    if len(mask.shape) == 4:
        mask = mask[0]

    bboxes = []
    mids = []
    for i in torch.unique(mask, sorted=True):
        if i == 0:
            continue
        xx, yy, zz = torch.where(mask == i)

        x1, y1, z1 = xx.min(), yy.min(), zz.min()
        x2, y2, z2 = xx.max(), yy.max(), zz.max()

        bboxes.append(torch.tensor([[x1, y1, z1], [x2, y2, z2]], dtype=torch.float32, device=mask.device))
        mids.append(i)

    device = mask.device
    if len(bboxes) == 0:
        bboxes = torch.zeros((0, 2, 3), dtype=torch.float32, device=device)
        mids = torch.zeros((0,), dtype=torch.float32, device=device)
    else:
        bboxes = torch.stack(bboxes, dim=0)
        mids = torch.stack(mids, dim=0)
    return bboxes, mids


import open3d as o3d
import seaborn



def vis_mask(mask, bboxes=None, mids=None, cls=None, mode=''):
    if len(mask.shape) == 4:
        mask = mask[0]

    if bboxes is None:
        bboxes, mids = extract_bbox_from_mask(mask)

    bboxes = bboxes.cpu().numpy()

    if mode == 'xyzwhd':
        center = bboxes[:, :3]
        whd = bboxes[:, 3:]
    elif mode == 'xyzxyz':
        pt1 = bboxes[:, :3]
        pt2 = bboxes[:, 3:]
        center = (pt1 + pt2) / 2
        whd = pt2 - pt1
    else:
        center = bboxes.mean(axis=1)
        whd = bboxes[:, 1] - bboxes[:, 0]

    o3d_vis = []
    mask = mask.cpu().numpy()
    colors = {}

    if mids is not None:
        mids = mids.cpu().numpy()
        count = 0
        for i, c, s in zip(mids, center, whd):
            i = int(i)

            if i not in colors:
                color = np.array(PALETTE[i])
                color = color.astype(np.float32) / 255
                # color = color[:, None]

                m = (mask == i).astype(np.uint8)
                mesh: trimesh.Trimesh = vol_to_mesh(m)
                mesh.visual.vertex_colors = color

                o3d_mesh: o3d.geometry.TriangleMesh = mesh.as_open3d
                o3d_mesh.compute_vertex_normals()
                o3d_mesh.paint_uniform_color(color)
                o3d_vis.append(o3d_mesh)
                colors[i] = color
            else:
                color = colors[i]

            count += 1

            vis_box = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=c - s / 2,
                max_bound=c + s / 2
            )
            vis_box.color = color
            o3d_vis.append(vis_box)
    else:

        for i in np.unique(mask):
            i = int(i)
            if i == 0:
                continue

            color = PALETTE[i]
            color = color.astype(np.float32) / 255
            color = color[:, None]
            m = (mask == i).astype(np.uint8)
            mesh: trimesh.Trimesh = vol_to_mesh(m)
            mesh.visual.vertex_colors = color

            o3d_mesh: o3d.geometry.TriangleMesh = mesh.as_open3d
            o3d_mesh.compute_vertex_normals()
            o3d_mesh.paint_uniform_color(color)
            o3d_vis.append(o3d_mesh)

        for c, s in zip(center, whd):
            color = PALETTE[i]
            color = color.astype(np.float32) / 255
            color = color[:, None]
            vis_box = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=c - s / 2,
                max_bound=c + s / 2
            )
            vis_box.color = color
            o3d_vis.append(vis_box)

        # b.colors = [color]
    o3d.visualization.draw_geometries(o3d_vis)


import imgaug as ia
from monai.data.box_utils import non_max_suppression


def show_3d(image, label, mode='multiclass'):
    if len(image.shape) == 4:
        image = image[0]
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()

    if mode == 'multilabel':
        c, h, w, z = label.shape
        bg = np.zeros((1, h, w, z), dtype=label.dtype)
        label = np.argmax(np.concatenate([bg, label], axis=0), axis=0)
    else:
        label = label[0]
        label = label % 40

    y, x, z = image.shape
    for i in range(z):
        frame = np.tile(image[..., i:i + 1], (1, 1, 3))
        frame_label = label[..., i].astype(np.uint8)

        frame = (frame * 255).astype(np.uint8)
        segmaps = ia.SegmentationMapsOnImage(frame_label, frame_label.shape)

        cv2.imshow('seg', frame)
        cv2.imshow('vis', segmaps.draw_on_image(frame, alpha=0.5, colors=PALETTE)[0][..., ::-1])
        cv2.waitKey()


def spiral_sphere(delta_angle=10., n=None):
    """
    n     angle diff (u, sigma)
    100   (18.7, 1.0)
    200   (13.3, 0.8)
    300   (10.8, 0.6)
    400   (9.3, 0.5)
    500   (8.5, 0.5)
    1000  (6, 0.3)
    2000  (4.2, 0.3)
    3000  (3.45, 0.22)
    4000  (3, 0.2)
    5000  (2.7, 0.2)
    6000  (2.44, 0.16)
    7000  (2.26, 0.147)
    10000 (1.9, 0.13)
    :param n:
    :return:
    """
    # estimate n
    delta_radius = delta_angle * np.pi / 180
    rhs = (1 - np.cos(delta_radius)) / (1 - np.cos(np.pi * (1 + 5 ** 0.5)))

    if n is None:
        n = int(1.5 / (1 - np.cos(np.arcsin(rhs ** 0.5))))

    # generate points on a sphere
    indices = np.arange(0, n, dtype=float) + 0.5

    theta = np.arccos(1 - 2 * indices / n)
    phi = np.pi * (1 + 5 ** 0.5) * indices

    x, y, z = np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)
    xyz = np.stack([x, y, z], axis=1)
    return xyz


def get_dst_spacing(spacing):
    spacing = np.array(spacing)
    max_spacing = np.max(spacing)
    min_spacing = np.min(spacing)

    if (max_spacing - min_spacing) > 0.4:
        dst_spacing = spacing
    elif min_spacing / max_spacing < 0.8:
        dst_spacing = np.array((max_spacing, max_spacing, max_spacing))
    else:
        dst_spacing = spacing

    return dst_spacing


def compute_size(spatial_size, spacing, dst_spacing, k=64, max_size=384, dst_size=None):
    spacing = np.abs(spacing)

    ratio = spacing / dst_spacing
    resize_shape = (spatial_size * ratio).astype(int)

    prod = resize_shape.prod()
    max_size_prod = max_size ** 3
    if prod > max_size_prod:
        resize_shape = resize_shape * ((max_size_prod / prod) ** (1 / 3))
        resize_shape = resize_shape.astype(int)

    if dst_size is not None:
        divisible_size = np.array([dst_size, dst_size, dst_size])
        ratio = (divisible_size / resize_shape).min()
        resize_shape = (resize_shape * ratio).astype(int)
    else:
        pass
        divisible_size = np.round(resize_shape / k).astype(int) * k
        resize_shape = np.where(resize_shape < divisible_size, resize_shape, divisible_size)

    pad_size = (divisible_size - resize_shape)
    pad_start = pad_size.astype(int) // 4 * 2
    pad_end = pad_size - pad_start
    pad_width = np.concatenate([pad_start[:, None], pad_end[:, None]], axis=1)
    pad_width = pad_width.flatten()
    return divisible_size, pad_width, resize_shape


import os
import json


def get_roi_from_labelme(case, vol_spacing, key):
    assert key in ['teeth', 'roi']

    labelme_dir = r'/home/wayne/data/cbct/original_data/slices_final'
    try:
        with open(os.path.join(labelme_dir, case + '_0.json')) as f:
            json1 = json.load(f)
            json1_dict = {s['label']: s['points'] for s in json1['shapes']}
        with open(os.path.join(labelme_dir, case + '_1.json')) as f:
            json2 = json.load(f)
            json2_dict = {s['label']: s['points'] for s in json2['shapes']}
    except Exception as e:
        print(case, e)

    (_, x1), (_, x2) = json1_dict[key]
    (z1, y1), (z2, y2) = json2_dict[key]
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    z1, z2 = sorted([z1, z2])

    if abs(vol_spacing[0] - vol_spacing[-1]) < 0.5:
        ratio = vol_spacing[0] / vol_spacing
        y1 *= ratio[0]
        y2 *= ratio[0]

        x1 *= ratio[1]
        x2 *= ratio[1]

        z1 *= ratio[2]
        z2 *= ratio[2]

    bbox = [y1, x1, z1, y2, x2, z2]
    return bbox


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou
