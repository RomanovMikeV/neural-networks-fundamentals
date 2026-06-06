import torch


class GoActorCritic(torch.nn.Module):
    def __init__(
            self,
            board_size=9,
            hidden_channels=(256, 128),
            activation=torch.nn.LeakyReLU):
        super().__init__()
        self.board_size = int(board_size)
        input_size = 3 * self.board_size * self.board_size
        hidden_channels = list(hidden_channels)

        self.backbone = torch.nn.Identity()
        self.critic = torch.nn.Linear(hidden_channels[-1], 1)
        self.actor = torch.nn.Linear(hidden_channels[-1], self.board_size * self.board_size)

        ## YOUR CODE HERE

    def __forward_kernel(self, board):
        features = board.reshape(board.shape[0], -1)
        ## YOUR CODE HERE
        return features

    def forward(self, batch):
        ## YOUR CODE HERE
        if 'signals' not in batch:
            features = batch['data']['board'].reshape(batch['data']['board'].shape[0], -1)
            batch['signals'] = {
                'features': features,
                'value': torch.zeros(features.shape[0], device=features.device),
                'move_logits': torch.zeros(
                    features.shape[0],
                    self.board_size * self.board_size,
                    device=features.device,
                ),
            }
            batch['postprocessed'] = {
                'move': torch.zeros(features.shape[0], dtype=torch.long, device=features.device),
            }
        return batch
