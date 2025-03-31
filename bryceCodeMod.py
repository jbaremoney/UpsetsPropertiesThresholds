
import torch
import torch.nn as nn

import torch.autograd as autograd
import math


K= 1.0
def signed_kaiming_constant_(tensor, a=0, mode='fan_in', nonlinearity='relu', k=1., sparsity =0):

    fan = nn.init._calculate_correct_fan(tensor, mode)  # calculating correct fan, depends on shape and type of nn
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = (gain / math.sqrt(fan))
    # scale by (1/sqrt(k))
    if k != 0:
        std *= (1 / math.sqrt(k))

    with torch.no_grad():
        tensor.uniform_(-std, std)
        if sparsity > 0:
            mask = (torch.rand_like(tensor) > sparsity).float()  # Keeps (1 - sparsity)% weights

            tensor *= mask
        return tensor

class GetSubnet(autograd.Function):

    @staticmethod
    def forward(ctx, scores, k):

        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1-k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, grad):

        # send the gradient g straight-through on the backward pass.
        return grad, None

# Our maskable replacement for the standard linear layer in torch.
# See the paper "What's Hidden in a Randomly Weighted Neural Network?" for
# more details (https://arxiv.org/abs/1911.13299)
# (this code adapted from https://github.com/iceychris/edge-popup)
class LinearSubnet(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, k=K, sparsity=0.5, zeroTrack=0.0,
                 init=signed_kaiming_constant_, **kwargs):
        super(LinearSubnet, self).__init__(in_features, out_features, bias=bias, **kwargs)
        self.k = k
        self.sparsity = sparsity
        self.zeroTrack = zeroTrack


        init(self.weight, k=k, sparsity=sparsity)

        # Save the full initial weight matrix for testing.
        self.initial_weight = self.weight.clone().detach()

        # Generate and store a full tensor of popup scores (one per weight)
        self.full_initial_popup_scores = torch.randn_like(self.weight).detach()

        # Create a tracking mask: always track nonzero weights.
        nonzero_mask = self.weight != 0
        if zeroTrack > 0:
            # For the zero weights, select a fraction to track.
            zero_mask = self.weight == 0
            # Get indices of zero weights (as a flat index for simplicity)
            flat_zero_indices = torch.nonzero(zero_mask.view(-1), as_tuple=False).view(-1)
            num_zero_to_track = int(zeroTrack * flat_zero_indices.numel())
            # Here we choose deterministically: the first num_zero_to_track indices.
            selected_zero_indices = flat_zero_indices[:num_zero_to_track]
            # Build a flat tracking mask (for all weights) that is True for:
            #  - all nonzero weights, and
            #  - the selected zero indices.
            flat_tracking_mask = nonzero_mask.view(-1).clone()
            # First, set all zero positions to False.
            flat_tracking_mask[torch.nonzero((self.weight.view(-1) == 0), as_tuple=False).view(-1)] = False
            # Now mark the selected zero positions as True.
            flat_tracking_mask[selected_zero_indices] = True
            # Reshape back to the weight shape.
            tracking_mask = flat_tracking_mask.view(self.weight.shape)
        else:
            tracking_mask = nonzero_mask

        # Save the tracking mask for later (and for testing)
        self.tracking_mask = tracking_mask

        # Initialize popup_scores as a parameter using the full initial values, but only for tracked weights
        self.popup_scores = nn.Parameter(self.full_initial_popup_scores[self.tracking_mask])
        # Also save these initial popup scores for later testing.
        self.initial_popup_scores = self.full_initial_popup_scores[self.tracking_mask].clone()


        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def update_zeroTrack(self, new_zeroTrack):
      """
      Update the popup_scores parameter based on a new zeroTrack value.
      This function recomputes the tracking mask and resets the popup_scores
      while keeping the original initializations for the positions that remain.

      Only really need this for testing. Basically just supposed to make it so
      we don't have to reinitialize a new network for each test. Not working as expected.
      Should decrease the number of trainable parameters when called to decrease zeroTrack.

      This version just gives an error about not
      """
      self.zeroTrack = new_zeroTrack
      nonzero_mask = self.initial_weight != 0
      if new_zeroTrack > 0:
        zero_mask = self.initial_weight == 0
        flat_zero_indices = torch.nonzero(zero_mask.view(-1), as_tuple=False).view(-1)
        num_zero_to_track = int(new_zeroTrack * flat_zero_indices.numel())
        selected_zero_indices = flat_zero_indices[:num_zero_to_track]
        flat_tracking_mask = nonzero_mask.view(-1).clone()
        flat_tracking_mask[torch.nonzero((self.initial_weight.view(-1) == 0), as_tuple=False).view(-1)] = False
        flat_tracking_mask[selected_zero_indices] = True
        tracking_mask = flat_tracking_mask.view(self.initial_weight.shape)
      else:
        tracking_mask = nonzero_mask

      self.tracking_mask = tracking_mask
      new_popup_init = self.full_initial_popup_scores[self.tracking_mask]

      # Properly replace the popup_scores parameter
      del self._parameters['popup_scores']
      self.register_parameter('popup_scores', nn.Parameter(new_popup_init.clone().detach(), requires_grad=True))

      self.initial_popup_scores = new_popup_init

      return self.popup_scores



class Network(nn.Module):
    def __init__(self, layer_sizes, maskable=False, k=K, sparsity=0., zeroTrack=0,init=signed_kaiming_constant_):
        super().__init__()
        self.flatten = nn.Flatten()

        # self.router = None
        if maskable:
            if isinstance(k, (int, float)): #MODIFIED to include the sparsity and zeroTrack
                self.linear_relu_stack = nn.Sequential(
                    *[m for m in [z for l in layer_sizes
                                  for z in [LinearSubnet(l[0], l[1], k=k, sparsity=sparsity, zeroTrack=zeroTrack), nn.ReLU()]][:-1]])
            else:
                self.linear_relu_stack = nn.Sequential(
                    *[m for m in [z for ind, l in enumerate(layer_sizes)
                                  for z in [LinearSubnet(l[0], l[1], k=k[ind], sparsity=sparsity, zeroTrack=zeroTrack), nn.ReLU()]][:-1]])
        else:
            self.linear_relu_stack = nn.Sequential(
                *[m for m in [z for l in layer_sizes for z in [nn.Linear(l[0], l[1]), nn.ReLU()]][:-1]])

    def count_nonzero_weights(self):
      total_nonzeros = 0
      for layer in self.linear_relu_stack:
        if hasattr(layer, 'weight'):  # Works for LinearSubnet or nn.Linear
            total_nonzeros += (layer.weight != 0).sum().item()
      return total_nonzeros

    def count_total_weights(self):
      total = 0
      for layer in self.linear_relu_stack:
        if hasattr(layer, 'weight'):
            total += layer.weight.numel()
      return total

    def count_total_popup_scores(model):
      total_scores = 0
      for layer in model.linear_relu_stack:
        if isinstance(layer, LinearSubnet) and hasattr(layer, 'popup_scores'):
            total_scores += layer.popup_scores.numel()
      return total_scores


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits