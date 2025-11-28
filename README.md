# HOPE: Nested Learning Reimplementation

This repository contains a PyTorch reimplementation of **HOPE** (High-order Optimization & Perception Engine)  , based on the paper [*"Nested Learning: The Illusion of Deep Learning"*](https://abehrouz.github.io/files/NL.pdf).

## Core Components

### 1. Nested Learning Paradigm

Unlike traditional Deep Learning which stacks layers, Nested Learning (NL) decomposes the architecture into "levels" based on update frequency.

  * **Code Location:** `src/utils/trainer.py`
  * **Implementation Note:** The training loop must handle **Multi-Time Scale Updates**. Parameters in different levels are updated with specific frequencies. You cannot use a standard `optimizer.step()` for all parameters every iteration. The trainer tracks the global step and only updates specific parameter groups (e.g., "High Frequency Neurons" vs "Low Frequency Neurons") when `step % frequency == 0`.

### 2. Continuum Memory System (CMS)

The CMS replaces the traditional Feed-Forward Network (FFN). It is a chain of MLP blocks where the $l$-th MLP is updated every $C^{(l)}$ steps.

  * **Code Location:** `src/memory/cms.py`
  * **Key Equation:** Implement Equation 30 for the forward pass and Equation 31 for the update logic.
  * **Structure:** A `nn.ModuleList` of MLPs. The forward pass is: $y_t = MLP^{(f_k)}(...MLP^{(f_1)}(x_t))$.

### 3. Self-Modifying Titans

This is the sequence mixing core. It uses a "Neural Learning Module" that learns to modify itself by learning its own update algorithm.

  * **Code Location:** `src/models/titans.py`
  * **Internal Optimizer:** The model uses a variant of gradient descent with regression loss as its internal forward pass mechanism.
  * **Key Equation:** The update rule for the weight $W_{t+1}$ follows Equation 28:
    $$W_{t+1} = W_t(I - x_t x_t^\top) - \eta_{t+1} \nabla_{y_t} \mathcal{L}(W_t; x_t) \otimes x_t$$.

### 4. Deep Optimizers

The paper argues that optimizers (like Adam or Momentum) are associative memory modules.

  * **Code Location:** `src/optimizers/deep_opt.py`
  * **Deep Momentum Gradient Descent (DMGD):** Implement Equation 23, where the momentum term is replaced by a neural network (e.g., an MLP) that compresses gradients.

## Configuration (`config/hope_config.yaml`)

You need to define the frequency hierarchy for the CMS layers.

```yaml
model:
  dim: 768
  depth: 12
  vocab_size: 50257

cms:
  # Define levels (Low to High frequency)
  # Frequencies are relative to the unit time step
  levels:
    - name: "level_1"
      frequency: 1       # Updates every step (High Freq)
      chunk_size: 16
    - name: "level_2"
      frequency: 16      # Updates every 16 steps
      chunk_size: 1000   # 1M in paper, scaled down for testing
    - name: "level_3"
      frequency: 1000    # Updates rarely (Low Freq)
      chunk_size: 0      # 0 implies rarely updated (Pre-training knowledge)
```

## Training

To train the HOPE model on Wikitext:

```bash
python train.py --config config/hope_config.yaml --dataset wikitext-103
```

## Key Architectural Differences

When implementing `src/models/hope.py`, ensure you distinguish it from a standard Transformer:

1.  **No fixed FFN:** Replace standard FFN blocks with the `CMS` module.
2.  **Dynamic Projections:** The $Q, K, V$ projections are not static; they are part of the optimization flow described in the "Self-Modifying Titans" section.
