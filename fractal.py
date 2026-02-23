import numpy as np
import torch
import random
import math

class SimplexNoise:
    def __init__(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        self.rng = np.random.default_rng(seed)
        
        # 初始化排列表 (Permutation Table)
        self.perm = self.rng.permutation(256)
        self.perm = np.concatenate([self.perm, self.perm])
        
        # 2D Simplex 梯度方向 (Gradients)
        self.grad3 = np.array([
            [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
            [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
            [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]
        ], dtype=np.float32)

    def noise2d(self, x, y):
        """
        生成 2D Simplex Noise。
        x, y: 形狀相同的 NumPy array (座標網格)
        """
        # Skew input space to determine which simplex cell we're in
        F2 = 0.5 * (np.sqrt(3.0) - 1.0)
        s = (x + y) * F2
        i = np.floor(x + s).astype(int)
        j = np.floor(y + s).astype(int)
        
        # Unskew back to (x,y) space
        G2 = (3.0 - np.sqrt(3.0)) / 6.0
        t = (i + j) * G2
        X0 = i - t
        Y0 = j - t
        x0 = x - X0
        y0 = y - Y0
        
        # Determine which simplex we are in
        i1 = (x0 > y0).astype(int)
        j1 = (x0 <= y0).astype(int)
        
        # Offsets for corners
        # x1 = x0 - i1 + G2
        # y1 = y0 - j1 + G2
        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1.0 + 2.0 * G2
        y2 = y0 - 1.0 + 2.0 * G2
        
        # Hash coordinates of the 3 simplex corners
        ii = i % 256
        jj = j % 256
        
        gi0 = self.perm[ii + self.perm[jj]] % 12
        gi1 = self.perm[ii + i1 + self.perm[jj + j1]] % 12
        gi2 = self.perm[ii + 1 + self.perm[jj + 1]] % 12
        
        # Calculate gradients
        # n = t^4 * dot(grad, dist)
        t0 = 0.5 - x0**2 - y0**2
        n0 = np.zeros_like(t0)
        mask0 = t0 >= 0
        if np.any(mask0):
            t0_m = t0[mask0]
            t0_m *= t0_m
            g0 = self.grad3[gi0[mask0]]
            dot0 = g0[:, 0] * x0[mask0] + g0[:, 1] * y0[mask0]
            n0[mask0] = (t0_m ** 2) * dot0
            
        t1 = 0.5 - x1**2 - y1**2
        n1 = np.zeros_like(t1)
        mask1 = t1 >= 0
        if np.any(mask1):
            t1_m = t1[mask1]
            t1_m *= t1_m
            g1 = self.grad3[gi1[mask1]]
            dot1 = g1[:, 0] * x1[mask1] + g1[:, 1] * y1[mask1]
            n1[mask1] = (t1_m ** 2) * dot1
            
        t2 = 0.5 - x2**2 - y2**2
        n2 = np.zeros_like(t2)
        mask2 = t2 >= 0
        if np.any(mask2):
            t2_m = t2[mask2]
            t2_m *= t2_m
            g2 = self.grad3[gi2[mask2]]
            dot2 = g2[:, 0] * x2[mask2] + g2[:, 1] * y2[mask2]
            n2[mask2] = (t2_m ** 2) * dot2

        # Sum up and scale to [-1, 1]
        return 70.0 * (n0 + n1 + n2)


class FractalNoise:
    """
    Fractal Noise Generator for Industrial Anomaly Synthesis.
    Supports:
    1. Fractal Brownian Motion (fBm) -> for Stains/Texture
    2. Ridged Multifractal -> for Scratches/Cracks
    """
    def __init__(self, size=(256, 256), seed=None):
        """
        Args:
            size: Tuple (height, width) of the output noise map.
            seed: Random seed.
        """
        self.h, self.w = size
        self.simplex = SimplexNoise(seed)
        
        # Pre-compute coordinate grid
        # Normalize coordinates to aspect ratio
        aspect = self.w / self.h
        y = np.linspace(0, 1, self.h)
        x = np.linspace(0, 1 * aspect, self.w)
        self.xv, self.yv = np.meshgrid(x, y)

    def generate_fbm(self, scale=4.0, octaves=4, persistence=0.5, lacunarity=2.0):
        """
        生成 Fractal Sum (fBm) 噪聲。
        適用於：表面異物、污漬 (Stains)、雲霧狀紋理。
        
        特性：連續性高，邊緣較平滑。
        """
        noise_sum = np.zeros((self.h, self.w), dtype=np.float32)
        amplitude = 1.0
        frequency = scale
        max_amplitude = 0.0
        
        for _ in range(octaves):
            # Generate noise layer
            n = self.simplex.noise2d(self.xv * frequency, self.yv * frequency)
            
            # Accumulate
            noise_sum += n * amplitude
            max_amplitude += amplitude
            
            # Prepare for next octave
            amplitude *= persistence
            frequency *= lacunarity
            
        # Normalize to [0, 1]
        noise_sum = (noise_sum / max_amplitude) + 0.5  # shift range roughly to [0, 1]
        noise_sum = np.clip(noise_sum, 0.0, 1.0)
        
        return noise_sum

    def generate_ridged(self, scale=10.0, octaves=4, persistence=0.5, lacunarity=2.0, power=2.0):
        """
        生成 Ridged Multifractal 噪聲。
        適用於：刮痕 (Scratches)、裂縫、細微紋路。
        
        原理：1 - |noise|。這會將原本的 0 值區域變成尖銳的山脊 (Ridge)。
        """
        noise_sum = np.zeros((self.h, self.w), dtype=np.float32)
        amplitude = 1.0
        frequency = scale
        max_amplitude = 0.0
        
        for _ in range(octaves):
            # Generate noise
            n = self.simplex.noise2d(self.xv * frequency, self.yv * frequency)
            
            # Ridged transformation: 1 - |n|
            # Simplex output is approx [-1, 1], so abs(n) is [0, 1]
            # 1 - abs(n) puts the 'zero-crossings' at 1 (peaks)
            n = 1.0 - np.abs(n)
            
            # Sharpen the ridges (optional, makes valleys wider and peaks thinner)
            if power != 1.0:
                n = np.power(n, power)
                
            noise_sum += n * amplitude
            max_amplitude += amplitude
            
            amplitude *= persistence
            frequency *= lacunarity
            
        # Normalize to [0, 1]
        noise_sum = noise_sum / max_amplitude
        noise_sum = np.clip(noise_sum, 0.0, 1.0)
        
        return noise_sum

    def get_mask(self, pattern_type='stain', binary_threshold=0.5, **kwargs):
        """
        統一介面取得 Mask。
        
        Args:
            pattern_type: 'stain' or 'scratch'
            binary_threshold: 若 > 0，則進行二值化處理回傳 0/1 mask。
            kwargs: 傳遞給 generate_fbm 或 generate_ridged 的參數
                    (如 scale, octaves, persistence...)
        """
        if pattern_type == 'scratch':
            # 預設刮痕參數：高頻率、較多細節
            params = {
                'scale': kwargs.get('scale', 12.0), # 較高的 scale 產生更細的紋路
                'octaves': kwargs.get('octaves', 4),
                'persistence': kwargs.get('persistence', 0.5),
                'lacunarity': kwargs.get('lacunarity', 2.0),
                'power': kwargs.get('power', 3.0) # 強化銳利度
            }
            noise_map = self.generate_ridged(**params)
            
        else: # stain
            # 預設污漬參數：低頻率、雲霧感
            params = {
                'scale': kwargs.get('scale', 4.0),
                'octaves': kwargs.get('octaves', 4), # 3-4 Octaves 增加邊緣隨機感
                'persistence': kwargs.get('persistence', 0.6),
                'lacunarity': kwargs.get('lacunarity', 2.0)
            }
            noise_map = self.generate_fbm(**params)

        if binary_threshold is not None:
            mask = (noise_map > binary_threshold).astype(np.float32)
            return mask
        
        return noise_map