from os import initgroups
import torch.nn as nn
import math
import torch
from torch import autograd
import torch.functional as F
# The default value of the parameter between 0 and 1 specifying the percentage
# of the model's weights that will be masked in each layer.
K = 1.0


# Set up signed Kaiming initialization.
def signed_kaiming_constant_(tensor, a=0, mode='fan_in', nonlinearity='sigmoid', k=1.):
    # fan_in means how many inputs coming into each neuron for layer
    fan = nn.init._calculate_correct_fan(tensor, mode)  # calculating correct fan, depends on shape and type of nn
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = (gain / math.sqrt(fan))
    # scale by (1/sqrt(k))
    if k != 0:
        std *= (1 / math.sqrt(k))
    with torch.no_grad():
        return tensor.uniform_(-std, std)


# A function to retreive a subset of the top k% of the weights by their score.
# The gradient is estimated by the identity (i.e. it goes "straight-through").
# See the paper "What's Hidden in a Randomly Weighted Neural Network?" for
# more details (https://arxiv.org/abs/1911.13299)
# (this code adapted from https://github.com/iceychris/edge-popup)
class GetSubnet(autograd.Function):
    # getSubnet is what generates our mask
    # so like GetSubnet applied to some network gives mask to apply to network
    @staticmethod
    def forward(ctx, scores, k):  # ctx saves tensor for backwards
        # scores is a tensor that tells you score for each weight
        # k tells u fraction of weights to keep

        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()  # copy of scores
        _, idx = scores.sort()  # sorting scores as 1d array
        # idx is sorted list of indexes of scores ascending
        # ie if index 2 has lowest it would be [2,...]
        # so we can use these indexes to get the right matrix indices for weight

        j = int((1 - k) * scores.numel())  # calculating num of weights to get rid

        # flat_out and out access the same memory.
        flat_out = out  # 1d array of score tensor
        flat_out[idx[:j]] = 0  # 0 out the weights in bottom 1-k%
        flat_out[idx[j:]] = 1  # keep the top k percent (mult by 1)

        return out

    @staticmethod
    def backward(ctx, grad):
        # send the gradient g straight-through on the backward pass.
        # grad is the just gradient of loss wrt forward pass
        # also returns none for k because not training k
        # maybe could train k?
        return grad, None


# Our maskable replacement for the standard linear layer in torch.
# See the paper "What's Hidden in a Randomly Weighted Neural Network?" for
# more details (https://arxiv.org/abs/1911.13299)
# (this code adapted from https://github.com/iceychris/edge-popup)
class LinearSubnet(nn.Linear):
    # inherits from nn.linear, meaning like fully connected layer but modified
    # we're adding the k aspect, popup scores

    def __init__(self, in_features, out_features, bias=True, k=K, init=signed_kaiming_constant_, **kwargs):
        # # of input neurons, # output neurons, bool:include bias, k from above, init method, extra arguments

        # calling parent constructor, makes fully connected layer
        super(LinearSubnet, self).__init__(in_features, out_features, bias if isinstance(bias, bool) else True,
                                           **kwargs)

        self.k = k
        # init weights
        if init == signed_kaiming_constant_:
            init(self.weight, k=k)
        else:
            init(self.weight)

        mask = self.weight != 0

        # Save only nonzero weight values
        self.weight_values = nn.Parameter(self.weight[mask])
        # outputs 1d tensor of nonzero weights

        # weight indices , row index tensor and column index tensor
        self.weight_indices = torch.nonzero(mask, as_tuple=True)  # store nonzero
        # (tensor([0, 0, 1, 1, 1, 2, 2]), tensor([1, 3, 0, 3, 4, 2, 4]))

        # initializing popup scores same shape as self.weight_values
        self.popup_scores = nn.Parameter(torch.randn_like(self.weight_values))

        self.initial_popup_scores = nn.Parameter(self.popup_scores.clone())  # Keep it trainable
        self.initial_weight = nn.Parameter(self.weight.clone())  # Keep it consistent

        # disable grad for the original parameters
        # not updating weights during backprop, just determining which weights
        # to use via popup scores
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off.
    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        adj = GetSubnet.apply(self.popup_scores.abs(), self.k)  # applying getSubnet
        # adj is the mask with only highest k% popup scores kept, others zeroed
        # PROBLEM: we sorted the popup scores. the corresponding indices didn't
        # get sorted with it
        # SOLVED: he used idx for list of indexes, which will preserve our indexes corresponding

        # Use only the subnetwork in the forward pass.
        # this masks the nonzero weights
        wPre = self.weight_values * adj

        w_sparse = torch.sparse_coo_tensor( #using chatgpt idea, slowed way down
            torch.stack(self.weight_indices),  # Stack row & col indices
            wPre,  # The values after pruning
            size=self.initial_weight.shape  # Shape of full weight matrix
        )

        return F.linear(x, w_sparse.to_dense(), self.bias)


# Our maskable replacement for the standard 2d convolutional layer in torch.
# See the paper "What's Hidden in a Randomly Weighted Neural Network?" for
# more details (https://arxiv.org/abs/1911.13299)
# (this code adapted from https://github.com/iceychris/edge-popup)
class Conv2dSubnet(nn.Conv2d):

    def __init__(self, *args, k=K, init=signed_kaiming_constant_, **kwargs):
        super(Conv2dSubnet, self).__init__(*args, **kwargs)
        self.k = k
        self.popup_scores = nn.Parameter(torch.randn(*self.weight.shape))

        # init weights
        init(self.weight, k=k)

        # disable grad for the original parameters
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off.
    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        adj = GetSubnet.apply(self.popup_scores.abs(), self.k)

        # Use only the subnetwork in the forward pass.
        w = self.weight * adj
        x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


# Our maskable replacement for the standard batchnorm2d layer in torch.
# See the paper "What's Hidden in a Randomly Weighted Neural Network?" for
# more details (https://arxiv.org/abs/1911.13299)
# (this code adapted from https://github.com/iceychris/edge-popup)
# class BatchNorm2dSubnet(nn.BatchNorm2d):

#     def __init__(self, *args, k=K, **kwargs):
#         super(nn.BatchNorm2d, self).__init__(*args, **kwargs)
#         self.k = k
#         self.popup_scores = nn.Parameter(torch.randn(*self.weight.shape))

#         # init weights
#         init(self.weight, k=k)

#         # disable grad for the original parameters
#         self.weight.requires_grad_(False)
#         if self.bias is not None:
#             self.bias.requires_grad_(False)

#     # self.k is the % of weights remaining, a real number in [0,1]
#     # self.popup_scores is a Parameter which has the same shape as self.weight
#     # Gradients to self.weight, self.bias have been turned off.
#     def forward(self, x):
#         # Get the subnetwork by sorting the scores.
#         adj = GetSubnet.apply(self.popup_scores.abs(), self.k)

#         # Use only the subnetwork in the forward pass.
#         w = self.weight * adj
#         x = F.batch_norm(x, self.running_mean, self.running_var, weight=w, bias=self.bias, momentum=0.1, eps=1e-05)
#         return x

# Our network builder, largely for convenience in testing.  It sets up a linear
# relu stack from a list of layer sizes.  It has an optional argument called
# "maskable" that, if true, will use our maskable linear layers instead of the
# stock linear layers in torch.
class Network(nn.Module):
    def __init__(self, layer_sizes, maskable=False, k=K, init=signed_kaiming_constant_):
        super().__init__()
        self.flatten = nn.Flatten()
        # self.router = None
        if maskable:
            if isinstance(k, (int, float)):
                self.linear_relu_stack = nn.Sequential(
                    *[m for m in [z for l in layer_sizes for z in [LinearSubnet(l[0], l[1], k=k), nn.ReLU()]][:-1]])
            else:
                self.linear_relu_stack = nn.Sequential(
                    *[m for m in [z for ind, l in enumerate(layer_sizes) for z in
                                  [LinearSubnet(l[0], l[1], k=k[ind], ), nn.ReLU()]][:-1]])
        else:
            self.linear_relu_stack = nn.Sequential(
                *[m for m in [z for l in layer_sizes for z in [nn.Linear(l[0], l[1]), nn.ReLU()]][:-1]])

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits