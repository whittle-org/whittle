from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: move this later to sth like train_fashion_mnist.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square conv kernel
        self.fc_base = torch.nn.Linear(28 * 28, 256)
        self.fc1 = nn.Linear(256, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward_dense_lottery(self, x, fc1_out, fc2_out, sample_last: Optional[bool] = False):
        x = x.reshape(x.shape[0], -1)
        x = self.fc_base(x)
        x = F.relu(x)
        if sample_last:
            start1 = 120 - fc1_out
            start2 = 84 - fc2_out
        else:
            start1 = 0
            start2 = 0
        # TODO: refactor this into a function since it is used in forward_dense_random_ind
        x = F.linear(x, weight=self.fc1.weight[start1:start1 + fc1_out, :], bias=self.fc1.bias[start1:start1 + fc1_out])
        x = F.relu(x)
        x = F.linear(x, weight=self.fc2.weight[start2:start2 + fc2_out, start1:start1 + fc1_out],
                     bias=self.fc2.bias[start2:start2 + fc2_out])
        x = F.relu(x)
        x = F.linear(x, weight=self.fc3.weight[:, start2:start2 + fc2_out], bias=self.fc3.bias)
        return x

    @staticmethod
    def sample_ids(max_id: int, num_ids: int):
        return np.random.choice(np.arange(max_id), num_ids)

    def forward_dense_random_neurons(self, x, fc1_out, fc2_out, seed=1234):
        """
        # TODO: replace these individual forwards with calls to lobotomy.modules.Linear 
        
        Randomly samples a subset of neurons from each layer (i.e., a sub-network) and computes the forward pass.
        Args:
            x: input tensor
            fc1_out: 
            fc2_out: 
            seed: 

        Returns:

        """
        np.random.seed(seed)
        x = x.reshape(x.shape[0], -1)
        # TODO: in the case of fc1_out=120 and fc2_out=84, this will sample all neurons, but currently with replacement
        fc1_ids = self.sample_ids(max_id=120, num_ids=fc1_out)
        fc2_ids = self.sample_ids(max_id=84, num_ids=fc2_out)

        x = self.fc_base(x)
        x = F.relu(x)
        x = F.linear(x, weight=self.fc1.weight[fc1_ids, :], bias=self.fc1.bias[fc1_ids])
        x = F.relu(x)
        x = F.linear(x, weight=self.fc2.weight[fc2_ids, :][:, fc1_ids], bias=self.fc2.bias[fc2_ids])
        x = F.relu(x)
        x = F.linear(x, weight=self.fc3.weight[:, fc2_ids], bias=self.fc3.bias)
        return x

    def forward_dense_rand_ind(self, x, fc1_out, fc2_out, start_id_fc1, start_id_fc2):
        x = x.reshape(x.shape[0], -1)
        x = self.fc_base(x)
        x = F.relu(x)
        x = F.linear(x, weight=self.fc1.weight[start_id_fc1:fc1_out + start_id_fc1, :],
                     bias=self.fc1.bias[start_id_fc1:start_id_fc1 + fc1_out])
        x = F.relu(x)
        x = F.linear(x,
                     weight=self.fc2.weight[start_id_fc2:start_id_fc2 + fc2_out, start_id_fc1:start_id_fc1 + fc1_out],
                     bias=self.fc2.bias[start_id_fc2:fc2_out + start_id_fc2])
        x = F.relu(x)
        x = F.linear(x, weight=self.fc3.weight[:, start_id_fc2:fc2_out + start_id_fc2], bias=self.fc3.bias)
        return x

    @staticmethod
    def sample_f_in():
        return np.random.randint(1, 121), np.random.randint(1, 85)

    @staticmethod
    def sample_f_in_and_ids():
        f1, f2 = np.random.randint(1, 121), np.random.randint(1, 85)
        start_id_f1 = np.random.randint(0, 121 - f1)
        start_id_f2 = np.random.randint(0, 85 - f2)
        return f1, f2, start_id_f1, start_id_f2

    def forward_sparse_lottery(self, x, fc1_out, fc2_out):
        if self.training:
            mask_fc1, mask_fc1_bias, mask_fc2, mask_fc2_bias, mask_fc3, mask_fc3_bias = self.sample_masks_sparse(
                fc1_out, fc2_out)
        else:
            mask_fc1, mask_fc1_bias, mask_fc2, mask_fc2_bias, mask_fc3, mask_fc3_bias = self.sample_masks_sparse_mag(
                fc1_out, fc2_out)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_base(x)
        x = F.relu(x)
        x = F.linear(x, weight=self.fc1.weight * mask_fc1, bias=self.fc1.bias * mask_fc1_bias)
        x = F.relu(x)
        x = F.linear(x, weight=self.fc2.weight * mask_fc2, bias=self.fc2.bias * mask_fc2_bias)
        x = F.relu(x)
        x = F.linear(x, weight=self.fc3.weight * mask_fc3, bias=self.fc3.bias * mask_fc3_bias)
        return x

    def forward_sparse_dense_lottery(self, x, fc1_out, fc2_out):
        weights_dense_fc1, weights_dense_fc2, weights_dense_fc3, bias_dense_fc1, bias_dense_fc2, bias_dense_fc3 = self.sample_sparse_dense_weights(
            fc1_out, fc2_out)
        # print(weights_dense_fc1)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_base(x)
        x = F.relu(x)
        x = F.linear(x, weight=weights_dense_fc1, bias=bias_dense_fc1)
        x = F.relu(x)
        x = F.linear(x, weight=weights_dense_fc2, bias=bias_dense_fc2)
        x = F.relu(x)
        x = F.linear(x, weight=weights_dense_fc3, bias=bias_dense_fc3)
        return x

    @staticmethod
    def sample_dense_dims():
        fc1_out = np.random.randint(1, 121)
        fc2_out = np.random.randint(1, 85)
        return fc1_out, fc2_out

    @staticmethod
    def sample_max():
        return 120, 84

    @staticmethod
    def sample_min():
        return 1, 1

    @staticmethod
    def sample_max_ids():
        return 120, 84, 0, 0

    @staticmethod
    def sample_min_ids():
        return 1, 1, np.random.randint(0, 121), np.random.randint(0, 85)

    def sample_masks_sparse(self, fc1_out, fc2_out):
        mask_fc1 = torch.zeros_like(self.fc1.weight)
        mask_fc1_bias = torch.zeros_like(self.fc1.bias)

        mask_fc2 = torch.zeros_like(self.fc2.weight)
        mask_fc2_bias = torch.zeros_like(self.fc2.bias)

        mask_fc3 = torch.zeros_like(self.fc3.weight)
        mask_fc3_bias = torch.zeros_like(self.fc3.bias)

        # TODO: understand what's happening here
        indices_fc1 = torch.randperm(mask_fc1.shape[0] * mask_fc1.shape[1])[:fc1_out * 256]
        indices_fc1 = torch.stack([indices_fc1 // 256, indices_fc1 % 256], dim=1)
        mask_fc1[indices_fc1[:, 0], indices_fc1[:, 1]] = 1

        indices_fc1_bias = torch.randperm(mask_fc1_bias.shape[0])[:fc1_out]
        mask_fc1_bias[indices_fc1_bias] = 1

        indices_fc2 = torch.randperm(mask_fc2.shape[0] * mask_fc2.shape[1])[:fc1_out * fc2_out]
        indices_fc2 = torch.stack([indices_fc2 // mask_fc2.shape[1], indices_fc2 % mask_fc2.shape[1]], dim=1)
        mask_fc2[indices_fc2[:, 0], indices_fc2[:, 1]] = 1

        indices_fc2_bias = torch.randperm(mask_fc2_bias.shape[0])[:fc2_out]
        mask_fc2_bias[indices_fc2_bias] = 1

        indices_fc3 = torch.randperm(mask_fc3.shape[0] * mask_fc3.shape[1])[:10 * fc2_out]
        indices_fc3 = torch.stack([indices_fc3 // mask_fc3.shape[1], indices_fc3 % mask_fc3.shape[1]], dim=1)
        mask_fc3[indices_fc3[:, 0], indices_fc3[:, 1]] = 1

        indices_fc3_bias = torch.randperm(mask_fc3_bias.shape[0])
        mask_fc3_bias[indices_fc3_bias] = 1

        return mask_fc1, mask_fc1_bias, mask_fc2, mask_fc2_bias, mask_fc3, mask_fc3_bias

    def sample_masks_sparse_mag(self, fc1_out, fc2_out):
        # Helper function to create a mask based on weight magnitude
        def create_mask(weight, n_ones):
            flattened_weights = weight.view(-1)
            threshold = torch.topk(flattened_weights.abs(), n_ones, largest=True).values[-1]
            mask = (weight.abs() >= threshold).float()
            return mask

        # Calculate number of ones for each mask
        n_ones_fc1 = fc1_out * 256
        n_ones_fc1_bias = fc1_out
        n_ones_fc2 = fc1_out * fc2_out
        n_ones_fc2_bias = fc2_out
        n_ones_fc3 = 10 * fc2_out
        n_ones_fc3_bias = 10

        # Create masks based on weight magnitude
        mask_fc1 = create_mask(self.fc1.weight, n_ones_fc1)
        mask_fc1_bias = create_mask(self.fc1.bias, n_ones_fc1_bias)
        mask_fc2 = create_mask(self.fc2.weight, n_ones_fc2)
        mask_fc2_bias = create_mask(self.fc2.bias, n_ones_fc2_bias)
        mask_fc3 = create_mask(self.fc3.weight, n_ones_fc3)
        mask_fc3_bias = create_mask(self.fc3.bias, n_ones_fc3_bias)

        return mask_fc1, mask_fc1_bias, mask_fc2, mask_fc2_bias, mask_fc3, mask_fc3_bias

    def sample_sparse_dense_weights(self, fc1_out, fc2_out):
        if self.training:
            mask_fc1, mask_fc1_bias, mask_fc2, mask_fc2_bias, mask_fc3, mask_fc3_bias = self.sample_masks_sparse(
                fc1_out, fc2_out)
        else:
            mask_fc1, mask_fc1_bias, mask_fc2, mask_fc2_bias, mask_fc3, mask_fc3_bias = self.sample_masks_sparse_mag(
                fc1_out, fc2_out)

        weights_dense_fc1 = self.fc1.weight[mask_fc1 != 0].reshape(fc1_out, 256)
        weights_dense_fc2 = self.fc2.weight[mask_fc2 != 0].reshape(fc2_out, fc1_out)
        weights_dense_fc3 = self.fc3.weight[mask_fc3 != 0].reshape(10, fc2_out)
        bias_dense_fc1 = self.fc1.bias[mask_fc1_bias != 0].reshape(fc1_out)
        bias_dense_fc2 = self.fc2.bias[mask_fc2_bias != 0].reshape(fc2_out)
        bias_dense_fc3 = self.fc3.bias[mask_fc3_bias != 0].reshape(10)
        return weights_dense_fc1, weights_dense_fc2, weights_dense_fc3, bias_dense_fc1, bias_dense_fc2, bias_dense_fc3

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc_base(x)
        x = F.relu(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = LeNet().to(device=device)
