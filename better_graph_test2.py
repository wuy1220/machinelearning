import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.draw import line, polygon, disk
import cv2

class OffshoreJacketDamageGenerator:
    """
    海洋导管架平台损伤图像生成器
    
    特点:
    - 程序化生成，速度快
    - 支持多种损伤类型
    - 可控损伤参数
    - 模拟海洋环境特征
    """
    
    def __init__(self, size=(512, 512), seed=None):
        self.size = size
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 导管架结构参数
        self.pipe_radius = 15  # 钢管半径（像素）
        self.node_positions = self._generate_jacket_structure()
        
        # 海洋环境参数
        self.sea_level_ratio = 0.5  # 海平面高度比例（0-1）
        self.splash_zone_width = 0.1  # 飞溅区宽度比例
        
    def _generate_jacket_structure(self):
        """生成导管架结构节点位置"""
        width, height = self.size
        nodes = []
        
        # 腿柱（4根）
        leg_positions = [
            (width * 0.25, height * 0.1),  # 左上前
            (width * 0.75, height * 0.1),  # 右上前
            (width * 0.2, height * 0.7),   # 左后下
            (width * 0.8, height * 0.7),   # 右后下
        ]
        
        # 中间节点（多水平层）
        num_levels = 5
        for level in range(num_levels):
            y_ratio = 0.15 + (level / (num_levels - 1)) * 0.6
            
            # 每层的中间节点（K型连接）
            for x_base_ratio in [0.3, 0.7]:
                x = width * x_base_ratio
                y = height * y_ratio
                
                # 添加透视效果（下窄上宽）
                perspective = 1.0 - (level / num_levels) * 0.1
                x = width * 0.5 + (x - width * 0.5) * perspective
                
                nodes.append((x, y))
        
        nodes.extend(leg_positions)
        return nodes
    
    def generate_base_structure(self):
        """生成基础导管架结构图像"""
        img = Image.new('RGB', self.size)
        draw = ImageDraw.Draw(img)
        
        # 1. 绘制海洋背景
        self._draw_ocean_background(img)
        
        # 2. 绘制钢管结构
        self._draw_jacket_structure(img, draw)
        
        # 3. 添加环境光照
        self._add_lighting_effects(img)
        
        return np.array(img).astype(np.float32) / 255.0
    
    def _draw_ocean_background(self, img):
        """绘制海洋环境背景"""
        width, height = self.size
        pixels = np.array(img).astype(np.float32)
        
        # 1. 天空渐变
        for y in range(int(height * 0.4)):
            sky_ratio = y / (height * 0.4)
            sky_color = (
                0.4 + 0.3 * sky_ratio,  # R
                0.5 + 0.3 * sky_ratio,  # G
                0.7 + 0.2 * sky_ratio   # B
            )
            pixels[y, :, :] = sky_color
        
        # 2. 海洋渐变（深到浅）
        for y in range(int(height * 0.4), height):
            sea_ratio = (y - height * 0.4) / (height * 0.6)
            sea_color = (
                0.1 + 0.1 * sea_ratio,  # R
                0.3 + 0.1 * sea_ratio,  # G
                0.5 + 0.2 * sea_ratio   # B
            )
            pixels[y, :, :] = sea_color
        
        # 3. 添加波浪纹理
        wave_layers = [
            {'y': int(height * 0.45), 'amplitude': 5, 'frequency': 0.05, 'color': (0.6, 0.7, 0.8)},
            {'y': int(height * 0.48), 'amplitude': 4, 'frequency': 0.08, 'color': (0.5, 0.6, 0.7)},
        ]
        
        for wave in wave_layers:
            y_wave = wave['y']
            amplitude = wave['amplitude']
            frequency = wave['frequency']
            color = wave['color']
            
            for x in range(width):
                wave_y = y_wave + amplitude * np.sin(2 * np.pi * frequency * x)
                wave_y = int(wave_y)
                if 0 <= wave_y < height:
                    # 添加波峰高光
                    for dy in range(-2, 3):
                        if 0 <= wave_y + dy < height:
                            alpha = 1.0 - abs(dy) / 3.0
                            pixels[wave_y + dy, x, :] = (
                                pixels[wave_y + dy, x, :] * (1 - alpha) +
                                np.array(color) * alpha
                            )
        
        # 4. 添加飞溅区和水位变动区标记
        sea_level = int(height * self.sea_level_ratio)
        splash_top = int(sea_level - height * self.splash_zone_width / 2)
        splash_bottom = int(sea_level + height * self.splash_zone_width / 2)
        
        # 飞溅区略亮的颜色
        for y in range(splash_top, splash_bottom):
            alpha = 0.1
            pixels[y, :, :] = pixels[y, :, :] * (1 - alpha) + np.array([0.6, 0.65, 0.7]) * alpha
        
        # 5. 添加水波纹
        for _ in range(20):
            cx, cy = random.randint(0, width), random.randint(int(height * 0.4), height)
            radius = random.randint(5, 15)
            
            y_grid, x_grid = np.ogrid[:height, :width]
            mask = ((x_grid - cx)**2 + (y_grid - cy)**2) <= radius**2
            mask_float = gaussian_filter(mask.astype(float), 2.0)
            
            # 水波高光
            for c in range(3):
                pixels[:, :, c] += mask_float * 0.05
        
        return Image.fromarray((np.clip(pixels, 0, 1) * 255).astype(np.uint8))
    
    def _draw_jacket_structure(self, img, draw):
        """绘制导管架钢管结构"""
        pixels = np.array(img).astype(np.float32)
        width, height = self.size
        
        # 定义连接关系（相邻节点连接）
        connections = []
        nodes_sorted = sorted(self.node_positions, key=lambda p: p[1])  # 按y坐标排序
        
        for i in range(len(nodes_sorted)):
            for j in range(i + 1, len(nodes_sorted)):
                x1, y1 = nodes_sorted[i]
                x2, y2 = nodes_sorted[j]
                
                # 只连接距离较近的节点
                distance = np.hypot(x2 - x1, y2 - y1)
                if distance < height * 0.25:
                    connections.append((nodes_sorted[i], nodes_sorted[j]))
        
        # 绘制钢管（带厚度）
        for (x1, y1), (x2, y2) in connections:
            # 管道主色（金属灰）
            pipe_color = np.array([0.5, 0.5, 0.5])
            
            # 计算线条角度和垂直方向
            dx = x2 - x1
            dy = y2 - y1
            angle = np.arctan2(dy, dx)
            
            # 绘制多条线形成管道
            num_lines = 2 * self.pipe_radius
            for i in range(num_lines):
                offset = i - num_lines / 2
                
                # 计算偏移坐标
                px1 = x1 + offset * (-np.sin(angle))
                py1 = y1 + offset * np.cos(angle)
                px2 = x2 + offset * (-np.sin(angle))
                py2 = y2 + offset * np.cos(angle)
                
                # 使用Bresenham算法
                rr, cc = line(int(py1), int(px1), int(py2), int(px2))
                
                # 确保在图像范围内
                valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                rr, cc = rr[valid], cc[valid]
                
                # 设置颜色（带阴影效果）
                shade = 1.0 - abs(offset) / self.pipe_radius * 0.3
                color = pipe_color * shade
                
                for c in range(3):
                    pixels[rr, cc, c] = color[c]
        
        # 绘制节点（球头）
        for x, y in self.node_positions:
            rr, cc = disk((y, x), self.pipe_radius + 2, shape=(height, width))
            
            # 节点颜色（深色）
            node_color = np.array([0.35, 0.35, 0.35])
            
            for c in range(3):
                pixels[rr, cc, c] = node_color[c]
            
            # 高光
            rr_highlight, cc_highlight = disk((y - 3, x - 3), 3, shape=(height, width))
            for c in range(3):
                pixels[rr_highlight, cc_highlight, c] += 0.2
        
        return Image.fromarray((np.clip(pixels, 0, 1) * 255).astype(np.uint8))
    
    def _add_lighting_effects(self, img):
        """添加光照效果"""
        pixels = np.array(img).astype(np.float32)
        width, height = self.size
        
        # 1. 添加全局光照渐变（模拟阳光）
        y_gradient = np.linspace(0.8, 1.0, height)[:, None]
        x_gradient = np.linspace(0.9, 1.0, width)[None, :]
        
        light_map = y_gradient * x_gradient
        
        for c in range(3):
            pixels[:, :, c] *= light_map
        
        # 2. 添加阴影（在结构下方）
        shadow_y = int(height * 0.35)
        for y in range(shadow_y, min(shadow_y + 20, height)):
            alpha = (y - shadow_y) / 20.0 * 0.2
            pixels[y, :, :] *= (1 - alpha)
        
        # 3. 添加镜面反射（在水面上方结构）
        reflect_y = int(height * 0.38)
        for y in range(max(0, reflect_y - 5), reflect_y + 5):
            alpha = 1.0 - abs(y - reflect_y) / 5.0
            pixels[y, :, :] += np.array([0.3, 0.35, 0.4]) * alpha * 0.1
        
        return Image.fromarray((np.clip(pixels, 0, 1) * 255).astype(np.uint8))
    
    # ========================================================================
    # 损伤生成方法
    # ========================================================================
    
    def add_corrosion_damage(self, img, damage_locations, severity=0.5):
        """
        添加腐蚀损伤
        
        参数:
            img: 基础图像
            damage_locations: 损伤位置列表 [(x, y, radius)]
            severity: 严重程度 (0.0-1.0)
        """
        pixels = np.array(img).astype(np.float32)
        height, width = self.size
        sea_level = int(height * self.sea_level_ratio)
        
        for x, y, radius in damage_locations:
            # 判断是否在水位变动区（腐蚀更严重）
            if abs(y - sea_level) < radius * 2:
                local_severity = min(1.0, severity * 1.5)
            else:
                local_severity = severity
            
            # 1. 生成锈色斑点
            num_spots = int(10 + 20 * local_severity)
            for _ in range(num_spots):
                # 斑点位置
                sx = x + np.random.uniform(-radius, radius)
                sy = y + np.random.uniform(-radius, radius)
                
                # 距离中心越近，斑点越大
                dist_to_center = np.hypot(sx - x, sy - y)
                spot_radius = np.random.uniform(2, 8) * (1 - dist_to_center / radius) * local_severity
                spot_radius = max(1, spot_radius)
                
                # 锈色（红褐色）
                rust_colors = [
                    np.array([0.6, 0.3, 0.1]),   # 深褐
                    np.array([0.7, 0.35, 0.15]), # 红褐
                    np.array([0.5, 0.25, 0.05]), # 暗褐
                    np.array([0.65, 0.2, 0.1]),  # 锈红
                ]
                rust_color = rust_colors[np.random.randint(0, len(rust_colors))]
                
                # 绘制斑点
                y_grid, x_grid = np.ogrid[:height, :width]
                mask = ((x_grid - sx)**2 + (y_grid - sy)**2) <= spot_radius**2
                
                # 边缘柔和
                mask_float = gaussian_filter(mask.astype(float), 1.5)
                
                # 应用腐蚀色
                for c in range(3):
                    pixels[:, :, c] += (rust_color[c] - pixels[:, :, c]) * mask_float * local_severity
            
            # 2. 添加剥落区域（局部漆皮脱落）
            if local_severity > 0.3:
                num_peel_areas = int(3 + 5 * (local_severity - 0.3))
                for _ in range(num_peel_areas):
                    px = x + np.random.uniform(-radius * 0.8, radius * 0.8)
                    py = y + np.random.uniform(-radius * 0.8, radius * 0.8)
                    peel_radius = np.random.uniform(5, 15) * local_severity
                    
                    # 剥落区域（露出金属原色）
                    y_grid, x_grid = np.ogrid[:height, :width]
                    mask = ((x_grid - px)**2 + (y_grid - py)**2) <= peel_radius**2
                    
                    # 边缘不规则
                    edge_noise = np.random.randn(*mask.shape) * 3
                    mask_float = gaussian_filter(mask.astype(float) + edge_noise, 2.0)
                    mask_float = np.clip(mask_float, 0, 1)
                    
                    # 金属原色（偏亮）
                    metal_color = np.array([0.7, 0.7, 0.75])
                    
                    for c in range(3):
                        pixels[:, :, c] = pixels[:, :, c] * (1 - mask_float) + metal_color[c] * mask_float
            
            # 3. 添加锈迹流痕（雨水/海浪冲刷痕迹）
            if local_severity > 0.5 and y < sea_level:
                # 从腐蚀点向下流
                for _ in range(np.random.randint(2, 5)):
                    start_x = x + np.random.uniform(-5, 5)
                    start_y = y
                    
                    # 流痕路径（随机波动）
                    path_x = [start_x]
                    path_y = [start_y]
                    
                    current_x = start_x
                    current_y = start_y
                    
                    while current_y < min(y + radius * 3, height - 1):
                        current_y += 1
                        current_x += np.random.uniform(-1, 1)
                        current_x = np.clip(current_x, 0, width - 1)
                        
                        path_x.append(current_x)
                        path_y.append(current_y)
                    
                    # 绘制流痕
                    for i in range(len(path_x) - 1):
                        x1, y1 = int(path_x[i]), int(path_y[i])
                        x2, y2 = int(path_x[i+1]), int(path_y[i+1])
                        
                        rr, cc = line(y1, x1, y2, x2)
                        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                        rr, cc = rr[valid], cc[valid]
                        
                        # 流痕颜色（红褐色，透明度渐减）
                        alpha = 1.0 - i / len(path_x)
                        stream_color = np.array([0.55, 0.3, 0.1])
                        
                        for c in range(3):
                            pixels[rr, cc, c] += (stream_color[c] - pixels[rr, cc, c]) * alpha * 0.3 * local_severity
        
        return Image.fromarray((np.clip(pixels, 0, 1) * 255).astype(np.uint8))
    
    def add_fatigue_crack_damage(self, img, damage_locations, severity=0.5):
        """
        添加疲劳裂纹损伤
        
        参数:
            img: 基础图像
            damage_locations: 损伤位置列表 [(x, y, length, angle)]
            severity: 严重程度 (0.0-1.0)
        """
        pixels = np.array(img).astype(np.float32)
        height, width = self.size
        
        for x, y, length, angle in damage_locations:
            # 生成主裂纹
            crack_points = self._generate_crack_path(x, y, length, angle, 
                                                    num_branches=int(2 + 3 * severity),
                                                    depth=int(2 + 2 * severity))
            
            # 绘制裂纹
            for segment in crack_points:
                x1, y1, x2, y2 = segment
                
                # 使用Bresenham绘制线条
                rr, cc = line(int(y1), int(x1), int(y2), int(x2))
                valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                rr, cc = rr[valid], cc[valid]
                
                # 裂纹宽度（根据严重程度）
                crack_width = max(1, int(1 + 2 * severity))
                
                # 绘制主裂纹（深色）
                for dw in range(-crack_width//2, crack_width//2 + 1):
                    offset = dw / (crack_width / 2)
                    alpha = 1.0 - abs(offset)
                    
                    # 横向偏移
                    rr_offset = np.clip(rr + dw * np.cos(angle), 0, height - 1).astype(int)
                    cc_offset = np.clip(cc + dw * np.sin(angle), 0, width - 1).astype(int)
                    
                    # 裂纹颜色（深黑色/深灰）
                    crack_color = np.array([0.0, 0.0, 0.0])
                    
                    for c in range(3):
                        pixels[rr_offset, cc_offset, c] = (
                            pixels[rr_offset, cc_offset, c] * (1 - alpha) +
                            crack_color[c] * alpha
                        )
            
            # 添加裂纹周围的应力集中区域（颜色略浅）
            for x, y, length, angle in damage_locations:
                stress_radius = int(10 + 20 * severity)
                y_grid, x_grid = np.ogrid[:height, :width]
                mask = ((x_grid - x)**2 + (y_grid - y)**2) <= stress_radius**2
                
                # 应力集中颜色（略亮）
                stress_color = np.array([0.6, 0.6, 0.65])
                alpha = 0.1 * severity
                
                for c in range(3):
                    pixels[:, :, c] += (stress_color[c] - pixels[:, :, c]) * mask * alpha
        
        return Image.fromarray((np.clip(pixels, 0, 1) * 255).astype(np.uint8))
    
    def _generate_crack_path(self, x0, y0, length, angle, num_branches=3, depth=3):
        """生成分形裂纹路径"""
        segments = []
        
        if depth == 0:
            return segments
        
        # 主裂纹
        x1 = x0 + np.cos(angle) * length
        y1 = y0 + np.sin(angle) * length
        segments.append((x0, y0, x1, y1))
        
        # 生成分支
        for _ in range(num_branches):
            branch_angle = angle + np.random.uniform(-np.pi/4, np.pi/4)
            branch_length = length * np.random.uniform(0.4, 0.7)
            
            sub_segments = self._generate_crack_path(
                x1, y1, branch_length, branch_angle,
                num_branches=max(1, num_branches - 1),
                depth=depth - 1
            )
            segments.extend(sub_segments)
        
        return segments
    
    def add_marine_fouling(self, img, coverage=0.3):
        """
        添加海洋生物附着（藤壶、贝类、藻类）
        
        参数:
            img: 基础图像
            coverage: 覆盖率 (0.0-1.0)
        """
        pixels = np.array(img).astype(np.float32)
        height, width = self.size
        sea_level = int(height * self.sea_level_ratio)
        
        # 只在水面以下添加
        num_organisms = int(20 + 80 * coverage)
        
        for _ in range(num_organisms):
            # 随机位置（水下）
            ox = random.randint(0, width - 1)
            oy = random.randint(sea_level + 10, height - 1)
            
            organism_type = random.choice(['barnacle', 'mussel', 'algae'])
            
            if organism_type == 'barnacle':
                # 藤壶（圆锥形）
                radius = random.randint(3, 8)
                height = random.randint(3, 7)
                
                # 绘制圆锥形
                for h in range(height):
                    current_radius = int(radius * (1 - h / height))
                    if current_radius > 0:
                        y_grid, x_grid = np.ogrid[:height, :width]
                        mask = ((x_grid - ox)**2 + (y_grid - (oy - h))**2) <= current_radius**2
                        
                        # 藤壶颜色（灰白）
                        barnacle_color = np.array([0.7, 0.7, 0.65])
                        alpha = 0.8
                        
                        for c in range(3):
                            pixels[:, :, c] = pixels[:, :, c] * (1 - mask * alpha) + barnacle_color[c] * mask * alpha
            
            elif organism_type == 'mussel':
                # 贻贝（扇形）
                num_mussels = random.randint(1, 4)
                for _ in range(num_mussels):
                    mx = ox + np.random.uniform(-15, 15)
                    my = oy + np.random.uniform(-5, 5)
                    length = random.randint(5, 12)
                    width = random.randint(3, 6)
                    angle = np.random.uniform(0, 2 * np.pi)
                    
                    # 贻贝颜色（深紫/黑色）
                    mussel_color = np.array([0.2, 0.15, 0.2])
                    
                    # 绘制椭圆（简化的贻贝）
                    y_grid, x_grid = np.ogrid[:height, :width]
                    # 旋转坐标
                    x_rot = (x_grid - mx) * np.cos(angle) + (y_grid - my) * np.sin(angle)
                    y_rot = -(x_grid - mx) * np.sin(angle) + (y_grid - my) * np.cos(angle)
                    
                    mask = ((x_rot / length)**2 + (y_rot / width)**2) <= 1.0
                    
                    for c in range(3):
                        pixels[:, :, c] = pixels[:, :, c] * (1 - mask * 0.9) + mussel_color[c] * mask * 0.9
            
            elif organism_type == 'algae':
                # 藻类（绿色丝状）
                num_strands = random.randint(3, 6)
                for _ in range(num_strands):
                    ax = ox + np.random.uniform(-10, 10)
                    ay = oy
                    
                    # 随机波动路径
                    points = [(ax, ay)]
                    current_x, current_y = ax, ay
                    
                    for _ in range(random.randint(20, 50)):
                        current_y -= 1  # 向上生长
                        current_x += np.random.uniform(-0.5, 0.5)
                        current_x = np.clip(current_x, 0, width - 1)
                        points.append((current_x, current_y))
                    
                    # 绘制藻类线条
                    for i in range(len(points) - 1):
                        x1, y1 = int(points[i][0]), int(points[i][1])
                        x2, y2 = int(points[i+1][0]), int(points[i+1][1])
                        
                        rr, cc = line(y1, x1, y2, x2)
                        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                        rr, cc = rr[valid], cc[valid]
                        
                        # 藻类颜色（绿色/褐色）
                        algae_color = np.array([0.2, 0.4, 0.2])
                        
                        for c in range(3):
                            pixels[rr, cc, c] = pixels[rr, cc, c] * 0.5 + algae_color[c] * 0.5
        
        return Image.fromarray((np.clip(pixels, 0, 1) * 255).astype(np.uint8))
    
    def add_dent_damage(self, img, damage_locations, severity=0.5):
        """
        添加凹陷损伤（船舶撞击）
        
        参数:
            img: 基础图像
            damage_locations: 损伤位置列表 [(x, y, width, height)]
            severity: 严重程度 (0.0-1.0)
        """
        pixels = np.array(img).astype(np.float32)
        height, width = self.size
        
        for x, y, dent_width, dent_height in damage_locations:
            # 1. 凹陷区域（颜色变化模拟阴影）
            y_grid, x_grid = np.ogrid[:height, :width]
            
            # 椭圆形凹陷
            mask = ((x_grid - x) / dent_width)**2 + ((y_grid - y) / dent_height)**2 <= 1.0
            
            # 边缘柔和
            mask_float = gaussian_filter(mask.astype(float), 3.0)
            
            # 凹陷颜色（较暗）
            dent_color = np.array([0.3, 0.3, 0.35])
            alpha = 0.5 * severity
            
            for c in range(3):
                pixels[:, :, c] = pixels[:, :, c] * (1 - mask_float * alpha) + dent_color[c] * mask_float * alpha
            
            # 2. 凹陷边缘高光（反射光）
            edge_mask = gaussian_filter(mask.astype(float), 2.0) - gaussian_filter(mask.astype(float), 4.0)
            edge_mask = np.maximum(edge_mask, 0)
            edge_mask = edge_mask / (edge_mask.max() + 1e-6)
            
            highlight_color = np.array([0.8, 0.8, 0.85])
            for c in range(3):
                pixels[:, :, c] += (highlight_color[c] - pixels[:, :, c]) * edge_mask * 0.3
            
            # 3. 刮痕（金属表面变形）
            if severity > 0.5:
                num_scratches = int(3 + 5 * (severity - 0.5))
                for _ in range(num_scratches):
                    sx = x + np.random.uniform(-dent_width/2, dent_width/2)
                    sy = y + np.random.uniform(-dent_height/2, dent_height/2)
                    s_length = np.random.uniform(5, 15)
                    s_angle = np.random.uniform(0, np.pi * 2)
                    
                    ex = sx + s_length * np.cos(s_angle)
                    ey = sy + s_length * np.sin(s_angle)
                    
                    rr, cc = line(int(sy), int(sx), int(ey), int(ex))
                    valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                    rr, cc = rr[valid], cc[valid]
                    
                    # 刮痕颜色（银色金属光泽）
                    scratch_color = np.array([0.6, 0.6, 0.7])
                    for c in range(3):
                        pixels[rr, cc, c] = pixels[rr, cc, c] * 0.7 + scratch_color[c] * 0.3
        
        return Image.fromarray((np.clip(pixels, 0, 1) * 255).astype(np.uint8))
    
    def add_camera_effects(self, img, blur_level=0.3, noise_level=0.05):
        """
        添加相机拍摄效果（模糊、噪声）
        
        参数:
            img: 基础图像
            blur_level: 模糊程度 (0.0-1.0)
            noise_level: 噪声程度 (0.0-1.0)
        """
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        
        # 1. 高斯模糊（模拟镜头散焦）
        if blur_level > 0:
            sigma = 0.5 + 2.0 * blur_level
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        # 2. 添加噪声
        pixels = np.array(img_pil).astype(np.float32) / 255.0
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, pixels.shape)
            pixels = np.clip(pixels + noise, 0, 1)
        
        # 3. 色差模拟（模拟相机镜头）
        if random.random() < 0.3:
            # 轻微色偏
            color_shift = np.random.uniform(-0.02, 0.02, 3)
            pixels = np.clip(pixels + color_shift, 0, 1)
        
        return Image.fromarray((np.clip(pixels, 0, 1) * 255).astype(np.uint8))
    
    def generate_complete_image(self, damage_config):
        """
        生成完整的损伤图像
        
        参数:
            damage_config: 损伤配置字典
                {
                    'corrosion': [{'location': (x, y, radius), 'severity': 0.6}],
                    'cracks': [{'location': (x, y, length, angle), 'severity': 0.7}],
                    'dents': [{'location': (x, y, width, height), 'severity': 0.5}],
                    'fouling': {'coverage': 0.3},
                    'camera': {'blur': 0.2, 'noise': 0.03}
                }
        """
        # 1. 生成基础结构
        img = self.generate_base_structure()
        
        # 2. 添加各种损伤
        if 'corrosion' in damage_config:
            for damage in damage_config['corrosion']:
                img = self.add_corrosion_damage(
                    img, 
                    [damage['location']], 
                    damage.get('severity', 0.5)
                )
        
        if 'cracks' in damage_config:
            for damage in damage_config['cracks']:
                img = self.add_fatigue_crack_damage(
                    img,
                    [damage['location']],
                    damage.get('severity', 0.5)
                )
        
        if 'dents' in damage_config:
            for damage in damage_config['dents']:
                img = self.add_dent_damage(
                    img,
                    [damage['location']],
                    damage.get('severity', 0.5)
                )
        
        if 'fouling' in damage_config:
            img = self.add_marine_fouling(
                img,
                coverage=damage_config['fouling'].get('coverage', 0.3)
            )
        
        # 3. 添加相机效果
        if 'camera' in damage_config:
            img = self.add_camera_effects(
                img,
                blur_level=damage_config['camera'].get('blur', 0.0),
                noise_level=damage_config['camera'].get('noise', 0.0)
            )
        
        return np.array(img).astype(np.float32) / 255.0

def main():
    damage_config = {
        'corrosion': [{'location': (0.2, 0.3, 0.1), 'severity': 0.6}],
        'cracks': [{'location': (0.4, 0.5, 0.2, 0.1), 'severity': 0.7}],
        'dents': [{'location': (0.6, 0.7, 0.1, 0.05), 'severity': 0.5}],
        'fouling': {'coverage': 0.3},
        'camera': {'blur': 0.2, 'noise': 0.03}
    }
    generator = OffshoreJacketDamageGenerator(size=(512, 512), seed=42)
    damaged_image = generator.generate_complete_image(damage_config)
    # 保存图像
    img_pil = Image.fromarray((damaged_image * 255).astype(np.uint8))
    img_pil.save('damaged_offshore_jacket.png')
    print("Damaged offshore jacket image saved as 'damaged_offshore_jacket.png'")

if __name__ == "__main__":
    main()