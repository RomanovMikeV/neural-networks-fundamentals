import numpy as np
import torch


class TinyNeRFDataset(torch.utils.data.Dataset):
    def __init__(self, image_size=24, n_samples=32, seed=0):
        self.image_size = int(image_size)
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.rays_o = []
        self.rays_d = []
        self.samples = []
        self.deltas = []
        self.target_rgb = []

        ## YOUR CODE HERE
        if not self.samples:
            zeros = np.zeros((self.n_samples, 2), dtype=np.float32)
            self.rays_o.append(np.zeros(2, dtype=np.float32))
            self.rays_d.append(np.array([0.0, 1.0], dtype=np.float32))
            self.samples.append(zeros)
            self.deltas.append(np.ones(self.n_samples, dtype=np.float32) / self.n_samples)
            self.target_rgb.append(np.zeros(3, dtype=np.float32))

    def _scene_density_and_color(self, points):
        centers = np.array([
            [-0.38, 0.05],
            [0.34, 0.18],
            [0.00, 0.56],
        ], dtype=np.float32)
        colors = np.array([
            [0.95, 0.20, 0.18],
            [0.15, 0.62, 0.95],
            [0.95, 0.82, 0.20],
        ], dtype=np.float32)
        widths = np.array([0.13, 0.16, 0.12], dtype=np.float32)
        density = np.zeros(points.shape[0], dtype=np.float32)
        weighted_color = np.zeros((points.shape[0], 3), dtype=np.float32)
        for center, color, width in zip(centers, colors, widths):
            distance2 = np.sum((points - center[None, :]) ** 2, axis=1)
            blob_density = 8.0 * np.exp(-distance2 / (2.0 * width ** 2))
            density += blob_density
            weighted_color += blob_density[:, None] * color[None, :]
        color = weighted_color / np.maximum(density[:, None], 1.0e-6)
        return density, color

    def _render_target(self, points, delta):
        density, color = self._scene_density_and_color(points)
        alpha = 1.0 - np.exp(-density * delta)
        transmittance = np.cumprod(np.concatenate([[1.0], 1.0 - alpha[:-1] + 1.0e-6]))
        weights = transmittance * alpha
        background = np.array([0.02, 0.03, 0.05], dtype=np.float32)
        rgb = np.sum(weights[:, None] * color, axis=0)
        rgb += np.maximum(0.0, 1.0 - np.sum(weights)) * background
        return np.clip(rgb, 0.0, 1.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {
            'ray_origin': torch.tensor(self.rays_o[index], dtype=torch.float32),
            'ray_direction': torch.tensor(self.rays_d[index], dtype=torch.float32),
            'samples': torch.tensor(self.samples[index], dtype=torch.float32),
            'deltas': torch.tensor(self.deltas[index], dtype=torch.float32),
            'target_rgb': torch.tensor(self.target_rgb[index], dtype=torch.float32),
        }
