import torch


class TinyNeRF(torch.nn.Module):
    def __init__(self, hidden_size=128, n_frequencies=6):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_frequencies = int(n_frequencies)
        encoded_size = 2 + 4 * self.n_frequencies

        self.field = torch.nn.Identity()
        self.density_head = torch.nn.Linear(encoded_size, 1)
        self.color_head = torch.nn.Linear(encoded_size, 3)

        ## YOUR CODE HERE

    def positional_encoding(self, coords):
        encoded = [coords]
        ## YOUR CODE HERE
        return torch.cat(encoded, dim=-1)

    def query_field(self, coords):
        encoded = self.positional_encoding(coords)
        features = self.field(encoded)
        density = torch.nn.functional.softplus(self.density_head(features)).squeeze(-1)
        color = torch.sigmoid(self.color_head(features))
        return density, color

    def render(self, samples, deltas):
        batch_size, n_samples, _ = samples.shape
        flat_samples = samples.reshape(batch_size * n_samples, 2)
        density, color = self.query_field(flat_samples)
        density = density.reshape(batch_size, n_samples)
        color = color.reshape(batch_size, n_samples, 3)

        alpha = 1.0 - torch.exp(-density * deltas)
        exclusive_transmittance = torch.cumprod(
            torch.cat([
                torch.ones(batch_size, 1, device=samples.device, dtype=samples.dtype),
                1.0 - alpha[:, :-1] + 1.0e-6,
            ], dim=1),
            dim=1,
        )
        weights = exclusive_transmittance * alpha
        rgb = (weights.unsqueeze(-1) * color).sum(dim=1)
        background = torch.tensor([0.02, 0.03, 0.05], device=samples.device, dtype=samples.dtype)
        rgb = rgb + (1.0 - weights.sum(dim=1, keepdim=True)).clamp_min(0.0) * background
        return rgb, weights, density, color

    def forward(self, batch):
        ## YOUR CODE HERE
        if 'signals' not in batch:
            batch_size = batch['data']['samples'].shape[0]
            n_samples = batch['data']['samples'].shape[1]
            device = batch['data']['samples'].device
            batch['signals'] = {
                'rgb': torch.zeros(batch_size, 3, device=device),
                'weights': torch.zeros(batch_size, n_samples, device=device),
                'density': torch.zeros(batch_size, n_samples, device=device),
                'sample_color': torch.zeros(batch_size, n_samples, 3, device=device),
            }
            batch['postprocessed'] = {'rgb': batch['signals']['rgb']}
        return batch
