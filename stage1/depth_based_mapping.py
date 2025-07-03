import numpy as np
import cv2
from typing import List, Tuple, Dict


def depth_based_3d_mapping(volume: np.ndarray, 
                            detection_2d: Dict,
                            axis: int,
                            threshold: float = -200) -> List[int]:
    """
    基于深度估计的2D到3D映射
    
    Args:
        volume: 3D体数据
        detection_2d: 2D检测结果 {"bbox": [x1,y1,x2,y2], "conf": 0.9}
        axis: 投影轴向 (0, 1, 2)
        threshold: CT值阈值，用于确定目标深度范围
        
    Returns:
        3D边界框 [z1,y1,x1,z2,y2,x2]
    """
    x1, y1, x2, y2 = detection_2d["bbox"]
    
    if axis == 0:  # 轴向0投影: (depth, height, width) -> (height, width)
        # 在检测区域内寻找目标深度范围
        roi = volume[:, y1:y2, x1:x2]
        # 找到高于阈值的体素
        mask = roi > threshold
        if np.any(mask):
            # 计算深度范围
            depth_coords = np.where(mask)[0]
            z1, z2 = depth_coords.min(), depth_coords.max()
            return [z1, y1, x1, z2, y2, x2]
            
    elif axis == 1:  # 轴向1投影: (depth, height, width) -> (depth, width)
        roi = volume[:, :, x1:x2]
        mask = roi > threshold
        if np.any(mask):
            depth_coords = np.where(mask)[0]
            height_coords = np.where(mask)[1]
            z1, z2 = depth_coords.min(), depth_coords.max()
            y1, y2 = height_coords.min(), height_coords.max()
            return [z1, y1, x1, z2, y2, x2]
            
    elif axis == 2:  # 轴向2投影: (depth, height, width) -> (depth, height)
        roi = volume[:, y1:y2, :]
        mask = roi > threshold
        if np.any(mask):
            depth_coords = np.where(mask)[0]
            width_coords = np.where(mask)[2]
            z1, z2 = depth_coords.min(), depth_coords.max()
            x1, x2 = width_coords.min(), width_coords.max()
            return [z1, y1, x1, z2, y2, x2]
    
    return None


def multi_axis_fusion(detections: List[Dict], 
                     volume: np.ndarray,
                     threshold: float = -200) -> List[Dict]:
    """
    融合多个轴向的检测结果
    
    Args:
        detections: 三个轴向的检测结果列表
        volume: 原始3D体数据
        threshold: CT值阈值
        
    Returns:
        融合后的3D边界框列表
    """
    bboxes_3d = []
    
    # 为每个检测结果计算3D边界框
    for axis, axis_detections in enumerate(detections):
        for det in axis_detections:
            bbox_3d = depth_based_3d_mapping(volume, det, axis, threshold)
            if bbox_3d is not None:
                bboxes_3d.append({
                    "bbox_3d": bbox_3d,
                    "conf": det["conf"],
                    "class": det["class"],
                    "axis": axis
                })
    
    # 合并重叠的边界框
    merged_bboxes = merge_overlapping_bboxes(bboxes_3d)
    
    return merged_bboxes


def merge_overlapping_bboxes(bboxes_3d: List[Dict], 
                           iou_threshold: float = 0.3) -> List[Dict]:
    """
    合并重叠的3D边界框
    
    Args:
        bboxes_3d: 3D边界框列表
        iou_threshold: IoU阈值
        
    Returns:
        合并后的边界框列表
    """
    if not bboxes_3d:
        return []
    
    # 计算3D IoU
    def calculate_3d_iou(bbox1, bbox2):
        z1_1, y1_1, x1_1, z2_1, y2_1, x2_1 = bbox1
        z1_2, y1_2, x1_2, z2_2, y2_2, x2_2 = bbox2
        
        # 计算交集
        z1_i = max(z1_1, z1_2)
        y1_i = max(y1_1, y1_2)
        x1_i = max(x1_1, x1_2)
        z2_i = min(z2_1, z2_2)
        y2_i = min(y2_1, y2_2)
        x2_i = min(x2_1, x2_2)
        
        if z1_i >= z2_i or y1_i >= y2_i or x1_i >= x2_i:
            return 0.0
        
        intersection = (z2_i - z1_i) * (y2_i - y1_i) * (x2_i - x1_i)
        volume1 = (z2_1 - z1_1) * (y2_1 - y1_1) * (x2_1 - x1_1)
        volume2 = (z2_2 - z1_2) * (y2_2 - y1_2) * (x2_2 - x1_2)
        union = volume1 + volume2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # 合并重叠的边界框
    merged = []
    used = [False] * len(bboxes_3d)
    
    for i in range(len(bboxes_3d)):
        if used[i]:
            continue
            
        current_group = [bboxes_3d[i]]
        used[i] = True
        
        for j in range(i + 1, len(bboxes_3d)):
            if used[j]:
                continue
                
            iou = calculate_3d_iou(bboxes_3d[i]["bbox_3d"], bboxes_3d[j]["bbox_3d"])
            if iou > iou_threshold:
                current_group.append(bboxes_3d[j])
                used[j] = True
        
        # 合并当前组
        if len(current_group) > 1:
            # 计算平均边界框
            avg_bbox = np.mean([bbox["bbox_3d"] for bbox in current_group], axis=0)
            avg_conf = np.mean([bbox["conf"] for bbox in current_group])
            merged.append({
                "bbox_3d": avg_bbox.astype(int).tolist(),
                "conf": avg_conf,
                "class": current_group[0]["class"],
                "merged_count": len(current_group)
            })
        else:
            merged.append(current_group[0])
    
    return merged


def example_usage():
    """使用示例"""
    # 模拟3D体数据
    volume = np.random.rand(512, 512, 262)
    
    # 模拟三个轴向的检测结果
    detections = [
        [{"bbox": [100, 150, 200, 250], "conf": 0.9, "class": 0}],  # 轴向0
        [{"bbox": [50, 150, 100, 200], "conf": 0.8, "class": 0}],   # 轴向1
        [{"bbox": [50, 100, 100, 250], "conf": 0.85, "class": 0}]   # 轴向2
    ]
    
    # 融合检测结果
    bboxes_3d = multi_axis_fusion(detections, volume)
    
    print("融合后的3D边界框:")
    for i, bbox_info in enumerate(bboxes_3d):
        print(f"目标 {i}: 3D边界框 {bbox_info['bbox_3d']}, 置信度 {bbox_info['conf']:.3f}")


if __name__ == "__main__":
    example_usage() 