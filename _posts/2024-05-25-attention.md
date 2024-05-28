---
layout: post
title: The detail about Attention structure
date: 2024-05-23 00:01:00
description: this is the detail about attention structure
tags: formatting code
categories: sample-posts
tabs: true
---

# Multi-Head Attention

## Scaled Dot-Product Attention and Multi-Head Attention

The attention architecture derives from the mechanisms of human attention. When an image is placed in front of a human, the human scans the global image to obtain areas that are worth focusing on, and devotes more attention to these areas to obtain information. Nowadays, the popular attention model generally relies on the encoder-decoder framework, which can deal with tasks including NLP, image processing, etc.

Scaled Dot-Product Attention is the most basic attention structure. Multi-Head Attention is multiple parallel Scaled Dot-Product Attention, which splices the output of each Scaled Dot-Product Attention and does a linear transformation to output.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/attention/scaledAndMultiAttention.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## Detail about the code of Multi-Head Attention

At the beginning, we need to import some libraries we will normally use.

```diff
import torch
from torch import nn
import torch.functional as F
import math
```

We need some data for testing. The dimension of testing data is 3. 

- The first dimension represents batch.
- The second dimension represents time. 
- The third dimension represnts the dimension after Encoder.

And in language model, the '512' below normally means the dimensions of the word vector do you want to map your word to after Embedding.

```diff
X = torch.randn(128,64,512) #Batch, Time, Dimension
print(X.shape)
```

Next, we set up the basic parameters of Multihead attention.
- d_model represents the number of dimensions I want to map to the QKV space.
- n_head repersents the number of head.

```diff
d_model = 512
n_head = 8
```

```diff
class multi_head_attention(nn.Module): # When we are writing a pytorch class, we need to inherit nn.Module
    def __init__(self, d_model, n_head) -> None:
        super(multi_head_attention, self).__init__() # Initialize some basic parameters

        self.d_model = d_model
        self.n_head = n_head

        self.w_q = nn.Linear(d_model, d_model) #Query
        self.w_k = nn.Linear(d_model, d_model) #Key
        self.w_v = nn.Linear(d_model, d_model) #Value
        # It's like we are looking for some queries to match some keys, and values are then weighted to combine.

        self.w_combine = nn.Linear(d_model, d_model)
        # Because of 'multi-head', we need a combinatorial mapping of the outputs.

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask = None):
        batch, time, dimension = q.shape # get the dimensions of Q, K, V
        n_d = self.d_model // self.n_head # Because we have n heads, 'd_model' for each submodel has to be divisible by 'n_head'.
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)   
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)   
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)   
        # We cannot put 'n_head' in the last dimension because we need to process the last two dimensions.

        score = q @ k.transpose(2, 3) / math.sqrt(n_d) # The most important code in Attention.

        minusInfty = -10000
        if mask is not None:
            score = score.masked_fill(mask == 0, minusInfty)
        score = self.softmax(score) @ v

        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)

        output = self.w_combine(score)
        
        return output 
```

```diff
attention = multi_head_attention(d_model, n_head)
output = attention(X, X, X)
print(output, output.shape)
```

## Something you might want to know about Attention


