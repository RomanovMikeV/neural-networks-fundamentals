import torch
import torch.nn.functional as F


class BinaryIndexTree:
    def __init__(self, vocab_size):
        self.vocab_size = int(vocab_size)
        if self.vocab_size <= 0:
            raise ValueError('vocab_size must be positive')
        self.depth = max(1, (self.vocab_size - 1).bit_length())
        self.max_path_length = self.depth
        self.num_internal_nodes = 2 ** self.depth - 1

    def targets_for_index(self, word_index):
        word_index = int(word_index)
        if word_index < 0 or word_index >= self.vocab_size:
            raise ValueError(f'word index {word_index} is outside vocabulary')
        return format(word_index, f'0{self.depth}b')

    @staticmethod
    def node_id_from_prefix(prefix_bits):
        if isinstance(prefix_bits, str):
            prefix = prefix_bits
        else:
            prefix = ''.join(str(int(bit)) for bit in prefix_bits)
        if not prefix:
            return 0
        depth = len(prefix)
        return (2 ** depth - 1) + int(prefix, 2)

    def path_and_targets(self, word_index):
        targets_string = self.targets_for_index(word_index)
        path = [
            self.node_id_from_prefix(targets_string[:step])
            for step in range(self.depth)
        ]
        targets = [int(bit) for bit in targets_string]
        return path, targets

    def __call__(self, context_word):
        device = context_word.device
        context_word = context_word.detach().cpu().view(-1).tolist()
        fallback_batch_size = len(context_word)
        # Fallback code for this task: keeps the notebook runnable before the method is solved.
        fallback = {
            'path': torch.zeros(
                fallback_batch_size,
                self.max_path_length,
                dtype=torch.long,
                device=device,
            ),
            'targets': torch.zeros(
                fallback_batch_size,
                self.max_path_length,
                dtype=torch.float32,
                device=device,
            ),
            'mask': torch.ones(
                fallback_batch_size,
                self.max_path_length,
                dtype=torch.float32,
                device=device,
            ),
        }

        ## YOUR CODE HERE
        return fallback


class HierarchicalSoftmax(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        # Fallback code for this task: creates a valid but deliberately weak decoder.
        self.embedding_dim = int(embedding_dim)
        self.targets = BinaryIndexTree(vocab_size)
        self.decoder = torch.nn.Embedding(
            self.targets.num_internal_nodes,
            self.embedding_dim,
        )
        torch.nn.init.zeros_(self.decoder.weight)

        ## YOUR CODE HERE

    @property
    def num_internal_nodes(self):
        return self.targets.num_internal_nodes

    @property
    def max_path_length(self):
        return self.targets.max_path_length

    def forward(self, embedding, target_word):
        target_tensors = self.targets(target_word)
        node_vectors = self.decoder(target_tensors['path'])
        fallback_logits = torch.einsum('bd,bld->bl', embedding, node_vectors) * 0.0
        fallback_probabilities = torch.sigmoid(fallback_logits)
        fallback_target_probabilities = torch.where(
            target_tensors['targets'].bool(),
            fallback_probabilities,
            1.0 - fallback_probabilities,
        )
        fallback_total_probability = fallback_target_probabilities.prod(dim=1)
        fallback_per_node_loss = F.binary_cross_entropy_with_logits(
            fallback_logits,
            target_tensors['targets'],
            reduction='none',
        )
        fallback_per_word_loss = (fallback_per_node_loss * target_tensors['mask']).sum(dim=1)
        # Fallback code for this task: returns valid tensors from neutral probabilities.
        fallback = {
            **target_tensors,
            'logits': fallback_logits,
            'probabilities': fallback_probabilities,
            'target_probabilities': fallback_target_probabilities,
            'total_probability': fallback_total_probability,
            'per_node_loss': fallback_per_node_loss,
            'per_word_loss': fallback_per_word_loss,
            'loss': fallback_per_word_loss.mean(),
        }

        ## YOUR CODE HERE
        return fallback


class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # Fallback code for this task: creates valid modules with weak zero embeddings.
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.encoder = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        torch.nn.init.zeros_(self.encoder.weight)
        self.hierarchical_softmax = HierarchicalSoftmax(
            self.embedding_dim,
            self.vocab_size,
        )
        self.decoder = self.hierarchical_softmax.decoder
        self.num_internal_nodes = self.hierarchical_softmax.num_internal_nodes

        ## YOUR CODE HERE


    def forward(self, batch):
        center_word = batch['data']['center_word']
        embedding = self.encoder(center_word)
        batch['signals'] = {
            'embedding': embedding,
        }
        batch['postprocessed'] = {}
        if 'context_word' in batch['data']:
            target_tensors = self.hierarchical_softmax.targets(batch['data']['context_word'])
            fallback_logits = torch.zeros_like(target_tensors['targets'])
            fallback_probabilities = torch.sigmoid(fallback_logits)
            fallback_target_probabilities = torch.where(
                target_tensors['targets'].bool(),
                fallback_probabilities,
                1.0 - fallback_probabilities,
            )
            fallback_per_node_loss = F.binary_cross_entropy_with_logits(
                fallback_logits,
                target_tensors['targets'],
                reduction='none',
            )
            fallback_per_word_loss = (fallback_per_node_loss * target_tensors['mask']).sum(dim=1)
            batch['data'].update(target_tensors)
            batch['signals']['logits'] = fallback_logits
            batch['signals']['probabilities'] = fallback_probabilities
            batch['signals']['target_probabilities'] = fallback_target_probabilities
            batch['signals']['total_probability'] = fallback_target_probabilities.prod(dim=1)
            batch['signals']['loss'] = fallback_per_word_loss.mean()
            batch['postprocessed']['targets'] = (fallback_probabilities >= 0.5).long()
        # Fallback code for this task: fills the batch with neutral predictions.
        fallback = batch

        ## YOUR CODE HERE
        return fallback
