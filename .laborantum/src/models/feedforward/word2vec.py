import heapq

import torch
import torch.nn.functional as F


class HierarchicalSoftmaxTargets:
    def __init__(self, word_counts, word_to_index):
        active_counts = {
            word_to_index[word]: int(count)
            for word, count in word_counts.items()
            if word in word_to_index
        }
        if len(active_counts) <= 1:
            self.paths = {0: [0]}
            self.codes = {0: [1]}
            self.num_internal_nodes = 1
            self.max_path_length = 1
            return

        next_internal_id = 0
        serial = 0
        heap = []
        for word_index, count in active_counts.items():
            heapq.heappush(heap, (count, serial, {'word': word_index}))
            serial += 1

        while len(heap) > 1:
            left_count, _, left = heapq.heappop(heap)
            right_count, _, right = heapq.heappop(heap)
            node = {
                'node': next_internal_id,
                'left': left,
                'right': right,
            }
            next_internal_id += 1
            heapq.heappush(heap, (left_count + right_count, serial, node))
            serial += 1

        paths = {}
        codes = {}

        def walk(node, path, code):
            if 'word' in node:
                paths[node['word']] = path.copy()
                codes[node['word']] = code.copy()
                return
            walk(node['left'], path + [node['node']], code + [0])
            walk(node['right'], path + [node['node']], code + [1])

        walk(heap[0][2], [], [])
        self.paths = paths
        self.codes = codes
        self.num_internal_nodes = next_internal_id
        self.max_path_length = max(len(path) for path in self.paths.values())

    def __call__(self, context_word):
        device = context_word.device
        context_word = context_word.detach().cpu().view(-1).tolist()
        paths = []
        codes = []
        masks = []
        for word_index in context_word:
            path = self.paths[int(word_index)]
            code = self.codes[int(word_index)]
            padding = self.max_path_length - len(path)
            paths.append(path + [0] * padding)
            codes.append(code + [0] * padding)
            masks.append([1.0] * len(path) + [0.0] * padding)
        return {
            'path': torch.tensor(paths, dtype=torch.long, device=device),
            'code': torch.tensor(codes, dtype=torch.float32, device=device),
            'mask': torch.tensor(masks, dtype=torch.float32, device=device),
        }


class HierarchicalSoftmaxLoss(torch.nn.Module):
    def __init__(self, model, targets):
        super().__init__()
        self.model = model
        self.targets = targets

    def forward(self, batch):
        target_tensors = self.targets(batch['data']['context_word'])
        batch['data'].update(target_tensors)
        embedding = batch['signals']['embedding']
        node_vectors = self.model.decoder(batch['data']['path'])
        logits = torch.einsum('bd,bld->bl', embedding, node_vectors)
        batch['signals']['logits'] = logits
        batch['signals']['probabilities'] = torch.sigmoid(logits)
        batch['postprocessed']['code'] = (batch['signals']['probabilities'] >= 0.5).long()
        per_node_loss = F.binary_cross_entropy_with_logits(
            logits,
            batch['data']['code'],
            reduction='none',
        )
        masked_loss = per_node_loss * batch['data']['mask']
        return masked_loss.sum() / batch['data']['mask'].sum().clamp_min(1.0)


class Word2VecHierarchicalSoftmax(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_internal_nodes):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.num_internal_nodes = int(num_internal_nodes)
        self.encoder = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.decoder = torch.nn.Embedding(self.num_internal_nodes, self.embedding_dim)

        ## YOUR CODE HERE

    def __forward_kernel(self, center_word, path):
        embedding = self.encoder(center_word)
        node_vectors = self.decoder(path)
        logits = torch.einsum('bd,bld->bl', embedding, node_vectors)
        return embedding, logits

    def forward(self, batch):
        ## YOUR CODE HERE
        if 'signals' not in batch:
            batch['signals'] = {
                'embedding': self.encoder(batch['data']['center_word']),
            }
            batch['postprocessed'] = {}
        return batch
