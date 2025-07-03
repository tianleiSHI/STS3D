import cv2
import numpy as np
from skimage import exposure
import nibabel as nib


def equalize(img):
    # 灰度百分位截断
    p2, p98 = np.percentile(img, (0.5, 99.5)) # 百分位截断，去除极值噪声
    p2, p98 = np.percentile(img, (2, 98)) # 医学影像中经常有极低值（空气）和极高值（金属），这些极值会影响对比度
    # 强度重映射
    img = exposure.rescale_intensity(img, in_range=(p2, p98), out_range=(0., 1.)) # 将像素值拉伸到0~1区间
    img = np.clip(img, 0, 1.) # 确保所有像素值都在0-1范围内
    # img = img.astype(np.int16)
    # Equalization
    # img_eq = exposure.equalize_hist(img) # 普通直方图均衡化
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.01) # 自适应直方图均衡化
    img_adapteq = (img_adapteq * 255).astype(np.uint8)
    return img_adapteq


# P123 noise
if __name__ == '__main__':
    import json
    import os
    import glob

    # 修改为你的CBCT数据文件夹路径
    data_dir = 'CBCT_data'
    
    # 查找所有.nii.gz文件
    nii_files = glob.glob(os.path.join(data_dir, '*.nii.gz'))
    
    if not nii_files:
        print(f"在 {data_dir} 中没有找到.nii.gz文件")
        exit()

    for nii_path in sorted(nii_files):
        print(f"处理文件: {os.path.basename(nii_path)}")
        
        # 使用nibabel读取NIfTI文件
        img = nib.load(nii_path)
        hu_img = img.get_fdata()  # 获取图像数据
        
        print(f"图像形状: {hu_img.shape}")
        print(f"数据类型: {hu_img.dtype}")
        print(f"数值范围: {hu_img.min():.2f} ~ {hu_img.max():.2f}")

        for i in [0, 1, 2]:
            stack_max = hu_img.max(axis=i)
            print(f"轴向 {i} 最大投影形状: {stack_max.shape}")

            eq_max = equalize(stack_max)
            
            # 显示图像
            cv2.imshow(f'Max Projection - Axis {i}', eq_max)
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()

