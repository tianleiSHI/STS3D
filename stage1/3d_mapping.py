import numpy as np
import cv2
from typing import List, Tuple, Dict
import json


class Detection3DMapper:
    """将2D检测结果映射回3D边界框的类"""
    
    def __init__(self, original_shape: Tuple[int, int, int]):
        """
        初始化映射器
        
        Args:
            original_shape: 原始3D数据的形状 (depth, height, width)
        """
        self.original_shape = original_shape
        self.depth, self.height, self.width = original_shape
        
    def map_2d_to_3d_bbox(self, 
                          detections_axis0: List[Dict],  # 轴向0的检测结果
                          detections_axis1: List[Dict],  # 轴向1的检测结果  
                          detections_axis2: List[Dict]   # 轴向2的检测结果
                          ) -> List[Dict]:
        """
        将三个方向的2D检测结果映射为3D边界框
        
        Args:
            detections_axis0: 轴向0的检测结果，格式 [{"bbox": [x1,y1,x2,y2], "conf": 0.9, "class": 0}]
            detections_axis1: 轴向1的检测结果
            detections_axis2: 轴向2的检测结果
            
        Returns:
            3D边界框列表，格式 [{"bbox_3d": [z1,y1,x1,z2,y2,x2], "conf": 0.9, "class": 0}]
        """
        # 轴向0: (depth, height, width) -> (height, width) 投影
        # 轴向1: (depth, height, width) -> (depth, width) 投影  
        # 轴向2: (depth, height, width) -> (depth, height) 投影
        
        bboxes_3d = []
        
        # 方法1: 基于投影几何关系的映射
        for det0 in detections_axis0:
            for det1 in detections_axis1:
                for det2 in detections_axis2:
                    # 检查是否为同一个目标（基于重叠度）
                    if self._is_same_object(det0, det1, det2):
                        bbox_3d = self._calculate_3d_bbox(det0, det1, det2)
                        if bbox_3d is not None:
                            bboxes_3d.append({
                                "bbox_3d": bbox_3d,
                                "conf": (det0["conf"] + det1["conf"] + det2["conf"]) / 3,
                                "class": det0["class"]
                            })
        
        return bboxes_3d
    
    def _is_same_object(self, det0: Dict, det1: Dict, det2: Dict) -> bool:
        """判断三个检测结果是否为同一个目标"""
        # 简单的启发式方法：检查类别是否相同，置信度是否都较高
        same_class = (det0["class"] == det1["class"] == det2["class"])
        high_conf = (det0["conf"] > 0.5 and det1["conf"] > 0.5 and det2["conf"] > 0.5)
        return same_class and high_conf
    
    def _calculate_3d_bbox(self, det0: Dict, det1: Dict, det2: Dict) -> List[int]:
        """
        根据三个方向的2D边界框计算3D边界框
        
        轴向0投影: (height, width) -> 提供 y1,y2,x1,x2
        轴向1投影: (depth, width) -> 提供 z1,z2,x1,x2  
        轴向2投影: (depth, height) -> 提供 z1,z2,y1,y2
        """
        try:
            # 从轴向0获取 y 和 x 坐标
            y1_0, x1_0, y2_0, x2_0 = det0["bbox"]
            
            # 从轴向1获取 z 和 x 坐标
            z1_1, x1_1, z2_1, x2_1 = det1["bbox"]
            
            # 从轴向2获取 z 和 y 坐标
            z1_2, y1_2, z2_2, y2_2 = det2["bbox"]
            
            # 融合坐标（取交集或平均值）
            x1 = max(x1_0, x1_1)
            x2 = min(x2_0, x2_1)
            y1 = max(y1_0, y1_2)
            y2 = min(y2_0, y2_2)
            z1 = max(z1_1, z1_2)
            z2 = min(z2_1, z2_2)
            
            # 检查边界框是否有效
            if x1 < x2 and y1 < y2 and z1 < z2:
                return [z1, y1, x1, z2, y2, x2]
            else:
                return None
                
        except Exception as e:
            print(f"计算3D边界框时出错: {e}")
            return None
    
    def visualize_3d_bbox(self, volume: np.ndarray, bboxes_3d: List[Dict], 
                         save_path: str = None):
        """可视化3D边界框（通过切片显示）"""
        for i, bbox_info in enumerate(bboxes_3d):
            bbox_3d = bbox_info["bbox_3d"]
            z1, y1, x1, z2, y2, x2 = bbox_3d
            
            # 显示中间切片
            mid_z = (z1 + z2) // 2
            mid_y = (y1 + y2) // 2
            mid_x = (x1 + x2) // 2
            
            # 三个正交切片
            slice_z = volume[mid_z, :, :]
            slice_y = volume[:, mid_y, :]
            slice_x = volume[:, :, mid_x]
            
            # 在切片上绘制边界框
            cv2.rectangle(slice_z, (x1, y1), (x2, y2), 255, 2)
            cv2.rectangle(slice_y, (x1, z1), (x2, z2), 255, 2)
            cv2.rectangle(slice_x, (y1, z1), (y2, z2), 255, 2)
            
            # 显示或保存
            if save_path:
                cv2.imwrite(f"{save_path}_bbox_{i}_z.png", slice_z)
                cv2.imwrite(f"{save_path}_bbox_{i}_y.png", slice_y)
                cv2.imwrite(f"{save_path}_bbox_{i}_x.png", slice_x)


def example_usage():
    """使用示例"""
    # 假设原始3D数据形状
    original_shape = (512, 512, 262)
    mapper = Detection3DMapper(original_shape)
    
    # 模拟2D检测结果（实际应该来自YOLO等模型）
    detections_axis0 = [
        {"bbox": [100, 150, 200, 250], "conf": 0.9, "class": 0}  # 牙齿类别
    ]
    detections_axis1 = [
        {"bbox": [50, 150, 100, 200], "conf": 0.8, "class": 0}
    ]
    detections_axis2 = [
        {"bbox": [50, 100, 100, 250], "conf": 0.85, "class": 0}
    ]
    
    # 映射到3D
    bboxes_3d = mapper.map_2d_to_3d_bbox(detections_axis0, detections_axis1, detections_axis2)
    
    print("3D边界框结果:")
    for i, bbox_info in enumerate(bboxes_3d):
        print(f"目标 {i}: 3D边界框 {bbox_info['bbox_3d']}, 置信度 {bbox_info['conf']:.3f}")


if __name__ == "__main__":
    example_usage() 