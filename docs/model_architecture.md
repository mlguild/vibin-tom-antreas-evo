# Model Architecture and Functional Forward Pass

This document details the model architecture (`SimpleResNet`) defined in `tinkering/models.py` and the utilities for functional and vectorized forward passes.

## 1. `SimpleResNet` Architecture

A custom ResNet-like architecture designed to be relatively lightweight and suitable for 28x28 single-channel images like EMNIST.

### 1.1. `BasicBlock`

Standard ResNet BasicBlock:
*   Two 3x3 `nn.Conv2d` layers with `nn.BatchNorm2d` and `nn.ReLU`.
*   Shortcut connection to enable residual learning. If input and output dimensions or strides differ, a 1x1 convolution is used in the shortcut to match dimensions.
*   `expansion = 1`: Output channels of the block are the same as the `planes` argument to its constructor.

### 1.2. `SimpleResNet` Class

*   **Initialization (`__init__`)**: 
    *   `block`: The block type to use (e.g., `BasicBlock`).
    *   `num_blocks_per_stage`: A list of 4 integers specifying the number of blocks in each of the 4 residual stages.
    *   `num_classes`: Number of output classes for the final linear layer.
    *   `in_channels`: Number of input channels for the image (default 1 for EMNIST).
*   **Initial Convolution (`conv1`, `bn1`)**: 
    *   `nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)` followed by `nn.BatchNorm2d(64)` and `nn.ReLU`. Stride is 1 to preserve resolution for small images.
*   **Four Residual Stages (`layer1` to `layer4`)**: 
    *   Built using the `_make_layer` helper method.
    *   The number of blocks in each stage is defined by `num_blocks_per_stage`.
    *   Channel progression: 64 -> 128 -> 256 -> 512.
    *   Downsampling (stride 2) occurs at the beginning of `layer2`, `layer3`, and `layer4` if the stride for the first block in those stages is 2.
*   **Pooling (`avgpool`)**: `nn.AdaptiveAvgPool2d((1, 1))` to reduce feature maps to 1x1 per channel.
*   **Classifier (`linear`)**: `nn.Linear(512 * block.expansion, num_classes)`.

### 1.3. `ResNet4StageCustom` Helper

```python
def ResNet4StageCustom(num_classes=10, in_channels=1):
    return SimpleResNet(
        BasicBlock,
        [1, 1, 1, 1], # One BasicBlock per stage
        num_classes=num_classes,
        in_channels=in_channels,
    )
```
This convenience function creates an instance of `SimpleResNet` with one `BasicBlock` in each of its four stages.

## 2. Functional and Vectorized Forward Passes

To support evolutionary algorithms where multiple model parameter sets need to be evaluated, the script utilizes `torch.func` utilities.

### 2.1. Core Functional Forward (`functional_forward_core`)

```python
def functional_forward_core(model_architecture, params_and_buffers, x_batch):
    return functional_call(model_architecture, params_and_buffers, (x_batch,))
```
*   This function takes a model *instance* (`model_architecture` for its structure), a tuple of `(parameters_dict, buffers_dict)`, and a single batch of data `x_batch`.
*   It performs a stateless forward pass using `torch.func.functional_call`.

### 2.2. Batched Models on a SINGLE Batch of Data (`get_parallel_forward_pass_fn`)

```python
def get_parallel_forward_pass_fn(model: nn.Module):
    def functional_forward_single_data(params_and_buffers, x):
        return functional_call(model, params_and_buffers, (x,))
    return vmap(functional_forward_single_data, in_dims=(0, None), randomness='same')
```
*   This higher-order function returns a new function that is vectorized using `torch.func.vmap`.
*   `in_dims=(0, None)` means:
    *   The first argument to `functional_forward_single_data` (which is `params_and_buffers`) will be mapped over along its 0th dimension (i.e., a batch of parameter/buffer sets).
    *   The second argument (`x`) will be broadcast (the same batch of data is used for every model in the population).

### 2.3. Batched Models on DIFFERENT Batches of Data

Demonstrated in `if __name__ == "__main__":` block of `tinkering/models.py`:

```python
# Assuming base_model is an instance of SimpleResNet
# params_and_buffers_for_vmap is a (stacked_params_dict, stacked_buffers_dict) tuple
# x_multiple_batches has shape [population_size, data_batch_size, C, H, W]

batched_fwd_multiple_data_fn = vmap(
    lambda p_b, x: functional_forward_core(base_model, p_b, x),
    in_dims=(0, 0), # Map over 0th dim of params_and_buffers AND 0th dim of x_multiple_batches
    randomness='same'
)
output = batched_fwd_multiple_data_fn(params_and_buffers_for_vmap, x_multiple_batches)
```
*   Here, `vmap` is configured with `in_dims=(0, 0)`. This means it will iterate through the population of model parameters/buffers AND simultaneously iterate through a batch of data batches.
*   The `i`-th model in the population processes the `i`-th batch of data from `x_multiple_batches`.
*   The output shape is `[population_size, data_batch_size, num_classes]`.

### 2.4. Preparing Batched Parameters/Buffers

`torch.func.stack_module_state(list_of_model_instances)` is used to create the batched parameter and buffer dictionaries required by `vmap` and `functional_call`.

1.  A list of model instances (e.g., `models_for_stacking`) is created, where each instance can have different weights (e.g., from a population in ES).
2.  `stacked_params_dict, stacked_buffers_dict = stack_module_state(models_for_stacking)`
3.  The `params_and_buffers_for_vmap` tuple `(stacked_params_dict, stacked_buffers_dict)` is then passed as the first argument to the `vmap`-ed function.

## 3. Test Block (`if __name__ == "__main__"`)

The script includes a comprehensive test block that demonstrates:
1.  Standard forward pass of a single model instance.
2.  Functional forward pass of a single model instance, asserting its output matches the standard pass.
3.  Creation of a population of model states (with slightly perturbed weights).
4.  Using `get_parallel_forward_pass_fn` to run the population on a single data batch.
5.  Using `vmap` with `in_dims=(0,0)` to run the population, each model on its own distinct data batch.

This ensures the core functionalities are working as expected. 