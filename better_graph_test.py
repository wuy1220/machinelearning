import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
from scipy.ndimage import gaussian_filter

def generate_enhanced_structure_image_v1(
    damaged_dofs=None, 
    noise_level=0.05,
    damage_subtlety=0.7,
    random_seed=None,
    add_texture=True
):
    """
    改进版1：增强颜色和纹理（接近真实结构照片）
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    # ======================================================================
    # 1. 生成结构背景（模拟混凝土/金属表面纹理）
    # ======================================================================
    target_size = 224
    
    # 创建基础画布
    img = Image.new('RGB', (target_size, target_size))
    pixels = np.array(img).astype(np.float32) / 255.0
    
    # ===== 添加混凝土纹理 =====
    # 生成随机噪声纹理
    base_texture = np.random.randn(target_size, target_size, 3) * 0.1
    
    # 使用不同尺度的高斯模糊创建多尺度纹理
    texture_multiscale = np.zeros_like(base_texture)
    for sigma in [1, 3, 8, 15]:
        texture_multiscale += gaussian_filter(np.random.randn(*base_texture.shape), sigma) * (1/sigma)
    
    # 归一化到合理范围
    texture_multiscale = (texture_multiscale - texture_multiscale.mean()) / (texture_multiscale.std() + 1e-6)
    texture_multiscale *= 0.15  # 控制纹理强度
    
    # 添加基础颜色变化（模拟不同老化程度的混凝土）
    # 主色调：灰色-浅褐色-浅黄色
    color_variations = [
        np.array([0.6, 0.6, 0.6]),   # 混凝土灰
        np.array([0.7, 0.65, 0.55]), # 水泥黄
        np.array([0.55, 0.55, 0.5]), # 深灰
    ]
    base_color = color_variations[random.randint(0, 2)]
    
    # 应用颜色和纹理
    pixels[:] = base_color + texture_multiscale
    
    # ===== 添加污渍和水渍 =====
    if random.random() < 0.7:
        # 随机生成污渍区域
        num_stains = random.randint(2, 5)
        for _ in range(num_stains):
            cx = random.randint(20, target_size-20)
            cy = random.randint(20, target_size-20)
            radius = random.randint(15, 40)
            
            y_grid, x_grid = np.ogrid[:target_size, :target_size]
            mask = ((x_grid - cx)**2 + (y_grid - cy)**2) <= radius**2
            
            # 边缘柔和的污渍
            edge_softness = random.uniform(5, 15)
            mask_float = gaussian_filter(mask.astype(float), edge_softness)
            mask_float = mask_float / mask_float.max()
            
            # 污渍颜色（深色或浅色）
            if random.random() < 0.5:
                stain_color = np.random.uniform(-0.15, -0.05, 3)  # 深色污渍
            else:
                stain_color = np.random.uniform(0.05, 0.15, 3)   # 浅色污渍
            
            pixels[mask] += stain_color * mask_float[mask][:, None]
    
    # 添加全局光照渐变（模拟阳光照射）
    if random.random() < 0.5:
        light_direction = random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])
        gradient = np.zeros((target_size, target_size))
        
        if 'top' in light_direction:
            y = np.linspace(1.0, 0.7, target_size)
            gradient = np.tile(y[:, None], (1, target_size))
        elif 'bottom' in light_direction:
            y = np.linspace(0.7, 1.0, target_size)
            gradient = np.tile(y[:, None], (1, target_size))
        
        if 'left' in light_direction:
            x = np.linspace(1.0, 0.8, target_size)
            gradient *= np.tile(x[None, :], (target_size, 1))
        elif 'right' in light_direction:
            x = np.linspace(0.8, 1.0, target_size)
            gradient *= np.tile(x[None, :], (target_size, 1))
        
        # 应用光照渐变
        for c in range(3):
            pixels[:, :, c] *= gradient
    
    # ======================================================================
    # 2. 绘制结构（节点和连接）
    # ======================================================================
    n_dof = 10
    margin = int(target_size * 0.15)
    draw_width = target_size - 2 * margin
    step = draw_width // (n_dof + 1)
    
    points = []
    for i in range(n_dof):
        x = margin + (i + 1) * step
        y = target_size // 2
        points.append((x, y))
        
        # 绘制节点（带阴影效果，增加立体感）
        node_radius = random.randint(5, 7)
        for dr in range(node_radius, 0, -1):
            alpha = 1.0 - (dr / node_radius) * 0.5
            color = np.array([0.15, 0.15, 0.15]) * alpha + base_color * (1 - alpha)
            
            y_grid, x_grid = np.ogrid[:target_size, :target_size]
            mask = ((x_grid - x)**2 + (y_grid - y)**2) <= dr**2
            
            # 软边
            mask_float = gaussian_filter(mask.astype(float), 1.0)
            valid = mask_float > 0.1
            
            for c in range(3):
                pixels[valid, c] = pixels[valid, c] * (1 - mask_float[valid]) + color[c] * mask_float[valid]
    
    # 绘制连接线（带不规则边缘）
    for i in range(len(points)-1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        
        # 使用Bresenham算法生成线条像素
        num_points = int(np.hypot(x2-x1, y2-y1)) + 1
        x_vals = np.linspace(x1, x2, num_points).astype(int)
        y_vals = np.linspace(y1, y2, num_points).astype(int)
        
        # 线条颜色（深灰色）
        line_color = np.array([0.25, 0.25, 0.25])
        line_width = random.randint(2, 4)
        
        for dw in range(-line_width, line_width+1):
            x_offset = np.clip(x_vals + dw * (y2-y1) / np.hypot(x2-x1, y2-y1), 0, target_size-1).astype(int)
            y_offset = np.clip(y_vals + dw * (x1-x2) / np.hypot(x2-x1, y2-y1), 0, target_size-1).astype(int)
            
            for k in range(num_points):
                # 距离衰减
                dist = abs(dw)
                intensity = max(0, 1 - dist / line_width)
                
                for c in range(3):
                    pixels[y_offset[k], x_offset[k], c] = pixels[y_offset[k], x_offset[k], c] * (1-intensity) + line_color[c] * intensity
    
    # ======================================================================
    # 3. 添加损伤（更真实的损伤表现）
    # ======================================================================
    if damaged_dofs:
        for dof in damaged_dofs:
            if dof < len(points):
                cx, cy = points[dof]
                damage_type = np.random.randint(0, 4)
                visibility_factor = max(0.05, 1.0 - damage_subtlety)
                
                if damage_type == 0:
                    # 腐蚀斑点（不规则形状）
                    num_spots = np.random.randint(3, 7)
                    for _ in range(num_spots):
                        spot_x = cx + np.random.uniform(-15, 15)
                        spot_y = cy + np.random.uniform(-15, 15)
                        spot_r = np.random.uniform(4, 12) * visibility_factor
                        
                        y_grid, x_grid = np.ogrid[:target_size, :target_size]
                        mask = ((x_grid - spot_x)**2 + (y_grid - spot_y)**2) <= spot_r**2
                        
                        # 边缘不规则化
                        edge_variation = np.random.randn(*mask.shape) * 2
                        mask_float = gaussian_filter(mask.astype(float), 2.0) + edge_variation * 0.3
                        mask_float = np.clip(mask_float, 0, 1)
                        
                        # 腐蚀颜色
                        corrosion_colors = [
                            np.array([-0.2, -0.15, -0.1]),  # 深褐色
                            np.array([-0.15, -0.1, -0.05]),  # 红褐色
                            np.array([-0.1, -0.08, -0.06]),  # 锈色
                        ]
                        color = corrosion_colors[np.random.randint(0, 3)]
                        
                        for c in range(3):
                            pixels[:, :, c] += color[c] * mask_float * 0.8
                
                elif damage_type == 1:
                    # 裂纹（分形分支）
                    num_cracks = np.random.randint(3, 6)
                    crack_colors = [
                        np.array([0.0, 0.0, 0.0]),      # 黑色
                        np.array([-0.1, -0.05, 0.0]),    # 深红褐
                        np.array([-0.05, -0.02, -0.02]), # 深灰
                    ]
                    color = crack_colors[np.random.randint(0, 3)]
                    
                    for _ in range(num_cracks):
                        # 分形生成裂纹
                        segments = _generate_fractal_crack(cx, cy, depth=3)
                        for (x1, y1, x2, y2) in segments:
                            # 绘制裂纹（带宽度变化）
                            num_points = int(np.hypot(x2-x1, y2-y1)) + 1
                            x_vals = np.linspace(x1, x2, num_points)
                            y_vals = np.linspace(y1, y2, num_points)
                            
                            # 沿着裂纹的宽度变化
                            width_profile = np.abs(np.sin(np.linspace(0, np.pi*2, num_points))) * 2 + 1
                            width_profile *= visibility_factor
                            
                            for k in range(num_points):
                                w = int(width_profile[k])
                                if w > 0:
                                    # 在裂纹周围添加影响
                                    y_grid, x_grid = np.ogrid[:target_size, :target_size]
                                    dist_sq = (x_grid - x_vals[k])**2 + (y_grid - y_vals[k])**2
                                    mask = dist_sq <= (w**2 + 2)
                                    
                                    intensity = np.exp(-dist_sq[mask] / (w**2 + 1))
                                    for c in range(3):
                                        pixels[mask, c] += color[c] * intensity
                
                elif damage_type == 2:
                    # 剥落（混凝土表面剥落）
                    peel_radius = np.random.uniform(12, 25) * visibility_factor
                    peel_depth = np.random.uniform(0.1, 0.25) * visibility_factor
                    
                    y_grid, x_grid = np.ogrid[:target_size, :target_size]
                    mask = ((x_grid - cx)**2 + (y_grid - cy)**2) <= peel_radius**2
                    
                    # 边缘不规则
                    mask_float = gaussian_filter(mask.astype(float), 3.0)
                    mask_float = mask_float / mask_float.max()
                    
                    # 剥落部分颜色（浅色，露出内部）
                    peel_color = np.array([0.8, 0.75, 0.7])
                    
                    for c in range(3):
                        pixels[:, :, c] = pixels[:, :, c] * (1 - mask_float) + peel_color[c] * mask_float
                    
                    # 剥落边缘阴影
                    edge_mask = gaussian_filter(mask.astype(float), 5.0) - gaussian_filter(mask.astype(float), 3.0)
                    edge_mask = np.maximum(edge_mask, 0)
                    edge_mask = edge_mask / (edge_mask.max() + 1e-6) * 0.3
                    
                    shadow_color = np.array([-0.1, -0.1, -0.08])
                    for c in range(3):
                        pixels[:, :, c] += shadow_color[c] * edge_mask
                
                elif damage_type == 3:
                    # 变形（结构几何改变）
                    deform_strength = np.random.uniform(0.5, 1.5) * visibility_factor
                    
                    # 节点偏移
                    y_grid, x_grid = np.ogrid[:target_size, :target_size]
                    dx = (x_grid - cx) * deform_strength * 0.02
                    dy = (y_grid - cy) * deform_strength * 0.02
                    
                    # 限制影响范围
                    radius = 25
                    mask = ((x_grid - cx)**2 + (y_grid - cy)**2) <= radius**2
                    mask_float = gaussian_filter(mask.astype(float), 5.0)
                    
                    # 变形效果（通过颜色变化模拟）
                    deform_intensity = np.abs(dx + dy) * 2.0
                    deform_intensity = deform_intensity * mask_float
                    
                    for c in range(3):
                        pixels[:, :, c] += deform_intensity * 0.2 * np.sign(np.random.randn())
    
    # ======================================================================
    # 4. 添加拍摄噪声（模拟真实照片）
    # ======================================================================
    # 高斯噪声（传感器噪声）
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, pixels.shape)
        pixels = pixels + noise
    
    # JPEG压缩伪影（模拟真实照片压缩）
    if random.random() < 0.3:
        img_temp = Image.fromarray((np.clip(pixels, 0, 1) * 255).astype(np.uint8))
        # 保存并重新加载（模拟JPEG压缩）
        import io
        buffer = io.BytesIO()
        quality = random.randint(70, 90)
        img_temp.save(buffer, format='JPEG', quality=quality)
        img_temp = Image.open(buffer)
        pixels = np.array(img_temp).astype(np.float32) / 255.0
    
    # ======================================================================
    # 5. 数据增强（训练时使用）
    # ======================================================================
    if add_texture:
        # 随机颜色抖动
        if random.random() < 0.5:
            brightness = np.random.uniform(0.9, 1.1)
            contrast = np.random.uniform(0.9, 1.1)
            pixels = np.clip((pixels - 0.5) * contrast + 0.5 * brightness, 0, 1)
        
        # 随机色温变化
        if random.random() < 0.3:
            color_shift = np.random.uniform(-0.05, 0.05, 3)
            pixels = np.clip(pixels + color_shift, 0, 1)
    
    # ======================================================================
    # 6. 裁剪和调整
    # ======================================================================
    pixels = np.clip(pixels, 0, 1)
    
    # 转换为 (C, H, W)
    return np.transpose(pixels, (2, 0, 1))

def _generate_fractal_crack(x0, y0, depth=3, length=20, angle_spread=np.pi/3):
    """生成分形裂纹"""
    if depth == 0:
        return [(x0, y0, x0 + np.random.uniform(-length/2, length/2), 
                 y0 + np.random.uniform(-length/2, length/2))]
    
    segments = []
    num_branches = np.random.randint(2, 4)
    
    for _ in range(num_branches):
        angle = np.random.uniform(-angle_spread/2, angle_spread/2)
        new_length = length * np.random.uniform(0.6, 0.9)
        
        dx = np.cos(angle) * new_length
        dy = np.sin(angle) * new_length
        
        x1 = x0 + dx
        y1 = y0 + dy
        
        segments.append((x0, y0, x1, y1))
        
        # 递归生成分支
        sub_segments = _generate_fractal_crack(x1, y1, depth-1, 
                                               new_length, angle_spread)
        segments.extend(sub_segments)
    
    return segments

if __name__ == '__main__':
    import os
    from datetime import datetime

    # 创建输出目录
    output_dir = 'generated_images'
    os.makedirs(output_dir, exist_ok=True)

    # 测试参数配置
    test_configs = [
        # 配置1: 无损伤
        {
            'name': 'no_damage',
            'params': {
                'damaged_dofs': None,
                'noise_level': 0.05,
                'damage_subtlety': 0.7,
                'random_seed': 42,
                'add_texture': True
            }
        },
        # 配置2: 单个损伤节点(腐蚀)
        {
            'name': 'single_damage_corrosion',
            'params': {
                'damaged_dofs': [3],
                'noise_level': 0.05,
                'damage_subtlety': 0.7,
                'random_seed': 42,
                'add_texture': True
            }
        },
        # 配置3: 多个损伤节点
        {
            'name': 'multiple_damages',
            'params': {
                'damaged_dofs': [2, 5, 8],
                'noise_level': 0.05,
                'damage_subtlety': 0.7,
                'random_seed': 42,
                'add_texture': True
            }
        },
        # 配置4: 低噪声
        {
            'name': 'low_noise',
            'params': {
                'damaged_dofs': [4],
                'noise_level': 0.02,
                'damage_subtlety': 0.7,
                'random_seed': 42,
                'add_texture': True
            }
        },
        # 配置5: 高噪声
        {
            'name': 'high_noise',
            'params': {
                'damaged_dofs': [4],
                'noise_level': 0.1,
                'damage_subtlety': 0.7,
                'random_seed': 42,
                'add_texture': True
            }
        },
        # 配置6: 损伤不明显
        {
            'name': 'subtle_damage',
            'params': {
                'damaged_dofs': [3, 7],
                'noise_level': 0.05,
                'damage_subtlety': 0.9,
                'random_seed': 42,
                'add_texture': True
            }
        },
        # 配置7: 损伤明显
        {
            'name': 'obvious_damage',
            'params': {
                'damaged_dofs': [3, 7],
                'noise_level': 0.05,
                'damage_subtlety': 0.3,
                'random_seed': 42,
                'add_texture': True
            }
        },
        # 配置8: 无纹理
        {
            'name': 'no_texture',
            'params': {
                'damaged_dofs': [4],
                'noise_level': 0.05,
                'damage_subtlety': 0.7,
                'random_seed': 42,
                'add_texture': False
            }
        }
    ]

    # 生成并保存测试图像
    print("开始生成测试图像...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for config in test_configs:
        print(f"正在生成: {config['name']}")

        # 生成图像
        image_data = generate_enhanced_structure_image_v1(**config['params'])

        # 转换为PIL图像
        image_data = np.transpose(image_data, (1, 2, 0))
        image_data = (np.clip(image_data, 0, 1) * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_data)

        # 保存图像
        filename = f"{config['name']}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        pil_image.save(filepath)
        print(f"已保存: {filepath}")

    print(f"\n所有测试图像已保存到 {output_dir} 目录")


