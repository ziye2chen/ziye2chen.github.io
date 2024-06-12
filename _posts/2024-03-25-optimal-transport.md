---
layout: distill
title: Optimal Transport
date: 2024-03-23 00:01:00
description: this is an introduction about Optimal Transport
tags: math
categories: sample-posts
tabs: true
featured: true

toc:
  - name: Intro to Optimal Transport
    subsections:
      - name: Measure
      - name: The Monge Formulation
      - name: The Kantorovich Formulation
      - name: Wasserstein Distance
    
  - name: Duality
    subsections:
      - name: Reminder of Dual Problem
      - name: Kantorovich Duality

  - name: Modern Solution of Optimal Transport (Sinkhorn)
    subsections:
      - name: Sinkhorn Distance
      - name: Computing Regularized Transport with Sinkhorn's Algorithm
      - name: PyTorch implementation of sinkhorn


_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }

---

## Intro to Optimal Transport

Optimal transport problem can be understood as a problem of two piles of soil, shoveled from soil A to the other and eventually piled up to form soil B. And your mission is to find the most efficient method to shovel the soil.

There are two ways to formulate the optimal transport problem: the Monge and Kantorovich formulations. Historically the Monge formulation comes before Kantorovich. The Kantorovich formulation can be seen as a generalisation of Monge. Both formulations have their advantages and disadvantages. Monge is more useful in applications, whilst Kantorovich is more useful theoretically. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/optimalTransport/soil.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### Measure

In mathematics, measure is a generalization and formalization of geometrical measures like length, area, volume, magnitude, mass and probability. In Lebesgue measure, for lower dimensions n = 1, 2, or 3, it coincides with the standard measure of length, area, or volume.

The concept of measures will be used later, and before I introduce probability measures, I need to introduce the histogram. The histogram here is the probability of n sums of 1, representing a probability distribution. We will use interchangeably the terms histogram and probability vector for any element $a \in \Sigma_n $ that belongs to the probability simplex

\begin{equation}
\Sigma_n := \{a \in \mathbb{R}_{+}^n : \sum_{i=1}^n a_i = 1  \}
\end{equation}

And we use discrete measure describes a probability measure if, additionally, $$a \in \Sigma_n $$ and more generally a positive measure if all the elements of vector $$a$$ are non-negative. 


A discrete measure with weights $$a$$ and locations $$x_1, \cdots, x_n \in \mathcal{X}$$ reads
\begin{equation}
\label{inter}
\alpha = \sum_{i=1}^n a_i \delta_{x_i},
\end{equation}
where $$\delta_x$$ is the Dirac at position x, intuitively a unit of mass which is infinitely concentrated at location $$x$$.
