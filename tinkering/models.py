import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap, stack_module_state
import copy
from rich.console import Console
from rich.pretty import pprint
from rich.traceback import install as install_rich_traceback

console = Console()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.InstanceNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = nn.InstanceNorm2d(planes, affine=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.InstanceNorm2d(self.expansion * planes, affine=True),
            )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        shortcut_out = self.shortcut(x)
        out += shortcut_out
        out = F.relu(out)
        return out


class SimpleResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks_per_stage,
        num_filters_base=4,
        num_classes=10,
        in_channels=1,
    ):
        super(SimpleResNet, self).__init__()
        self.in_planes = 16

        # Initial convolution: Modified for small images (e.g., 28x28) and single channel
        self.conv1 = nn.Conv2d(
            in_channels,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.InstanceNorm2d(16, affine=True)

        # Four stages of residual blocks
        self.layer1 = self._make_layer(
            block, num_filters_base, num_blocks_per_stage[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, num_filters_base * 2, num_blocks_per_stage[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, num_filters_base * 4, num_blocks_per_stage[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, num_filters_base * 8, num_blocks_per_stage[3], stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(
            num_filters_base * 8 * block.expansion, num_classes
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def ResNet4StageCustom(num_classes=10, in_channels=1):
    """Constructs a ResNet with 4 stages, each containing one BasicBlock."""
    return SimpleResNet(
        BasicBlock,
        [1, 1, 1, 1],
        num_classes=num_classes,
        in_channels=in_channels,
    )


# Helper for batched models on a SINGLE batch of data
def get_parallel_forward_pass_fn(model: nn.Module):
    def functional_forward_single_data(params_and_buffers, x):
        return functional_call(model, params_and_buffers, (x,))

    # in_dims=(0, None) -> map over params_and_buffers, use same x for all
    return vmap(
        functional_forward_single_data, in_dims=(0, None), randomness="same"
    )


# Core functional forward for one model, one data batch (to be vmapped for various scenarios)
def functional_forward_core(model_architecture, params_and_buffers, x_batch):
    return functional_call(model_architecture, params_and_buffers, (x_batch,))


if __name__ == "__main__":
    install_rich_traceback()
    console.rule("[bold green]Demonstrating Model Forward Passes[/bold green]")

    # --- Common Setup ---
    num_emnist_classes = 62
    data_batch_size = 4  # Batch size for a single batch of data
    population_size = 3  # Number of models in our population
    C, H, W = 1, 28, 28  # EMNIST-like image dimensions

    console.print(
        f"[info]Common params: num_classes={num_emnist_classes}, data_batch_size={data_batch_size}, population_size={population_size}"
    )

    # --- Example 1a: Standard PyTorch Model Forward Pass ---
    console.rule("[bold blue]Ex 1a: Standard Model Forward Pass[/bold blue]")
    base_model = ResNet4StageCustom(
        num_classes=num_emnist_classes, in_channels=C
    )
    console.print("Base model structure:")
    # pprint(base_model) # Can be verbose, uncomment if needed
    x_single_batch = torch.randn(data_batch_size, C, H, W)
    console.print(f"Input data shape (single batch): {x_single_batch.shape}")
    output_standard = base_model(x_single_batch)
    console.print(f"Standard output shape: {output_standard.shape}")

    # --- Example 1b: Single Model Functional Forward Pass (Non-Batched, Non-Vmapped) ---
    console.rule(
        "[bold blue]Ex 1b: Single Model Functional Forward Pass[/bold blue]"
    )
    # Extract parameters and buffers for the base_model
    params_dict = {
        name: param for name, param in base_model.named_parameters()
    }
    buffers_dict = {
        name: buffer for name, buffer in base_model.named_buffers()
    }

    output_functional_single = functional_forward_core(
        base_model, (params_dict, buffers_dict), x_single_batch
    )
    console.print(
        f"Functional single model output shape: {output_functional_single.shape}"
    )
    assert torch.allclose(
        output_standard, output_functional_single, atol=1e-6
    ), "Standard and functional outputs differ!"
    console.print(
        "[success]Standard and functional single model outputs match."
    )

    # --- Prepare a population of model states (parameters and buffers) ---
    console.print(
        f"\n[info]Creating a population of {population_size} model states..."
    )
    models_for_stacking = []
    for i in range(population_size):
        perturbed_model = ResNet4StageCustom(
            num_classes=num_emnist_classes, in_channels=C
        )
        perturbed_model.load_state_dict(
            base_model.state_dict()
        )  # Start with base model weights
        with torch.no_grad():
            for param in perturbed_model.parameters():
                param.add_(torch.randn_like(param) * 0.01)  # Add small noise
        models_for_stacking.append(perturbed_model)

    stacked_params_dict, stacked_buffers_dict = stack_module_state(
        models_for_stacking
    )
    # `params_and_buffers_for_vmap` is a tuple where each element is a dict of batched tensors
    params_and_buffers_for_vmap = (stacked_params_dict, stacked_buffers_dict)
    console.print(
        f"[info]Stacked {len(stacked_params_dict)} parameter tensors and {len(stacked_buffers_dict)} buffer tensors."
    )
    # Example: check shape of one stacked param
    # first_param_name = next(iter(stacked_params_dict))
    # console.print(f"  Shape of a stacked param ('{first_param_name}'): {stacked_params_dict[first_param_name].shape}")

    # --- Example 2: Batched Models (Weights) on a SINGLE Batch of Data (Using the Helper) ---
    console.rule(
        "[bold blue]Ex 2: Batched Models on Single Data Batch[/bold blue]"
    )
    batched_fwd_single_data_fn = get_parallel_forward_pass_fn(base_model)
    console.print(f"Using input data shape: {x_single_batch.shape}")
    output_batched_models_single_data = batched_fwd_single_data_fn(
        params_and_buffers_for_vmap, x_single_batch
    )
    console.print(
        f"Batched models, single data batch output shape: {output_batched_models_single_data.shape}"
    )
    expected_shape_ex2 = (population_size, data_batch_size, num_emnist_classes)
    assert (
        output_batched_models_single_data.shape == expected_shape_ex2
    ), f"Expected shape {expected_shape_ex2}, got {output_batched_models_single_data.shape}"
    console.print(f"[success]Output shape is correct: {expected_shape_ex2}")

    # --- Example 3: Batched Models (Weights) on DIFFERENT Batches of Data (Batch of Batches) ---
    console.rule(
        "[bold blue]Ex 3: Batched Models on Different Data Batches[/bold blue]"
    )
    # Each of the {population_size} models gets its own batch of {data_batch_size} images.
    x_multiple_batches = torch.randn(population_size, data_batch_size, C, H, W)
    console.print(
        f"Input data shape (batch of batches): {x_multiple_batches.shape}"
    )

    # Vectorize the core functional forward for this scenario: in_dims=(0, 0)
    # 0 for params_and_buffers (map over models)
    # 0 for x_batch (map over data batches)
    batched_fwd_multiple_data_fn = vmap(
        lambda p_b, x: functional_forward_core(base_model, p_b, x),
        in_dims=(0, 0),
        randomness="same",
    )

    output_batched_models_multiple_data = batched_fwd_multiple_data_fn(
        params_and_buffers_for_vmap, x_multiple_batches
    )
    console.print(
        f"Batched models, multiple data batches output shape: {output_batched_models_multiple_data.shape}"
    )
    expected_shape_ex3 = (population_size, data_batch_size, num_emnist_classes)
    assert (
        output_batched_models_multiple_data.shape == expected_shape_ex3
    ), f"Expected shape {expected_shape_ex3}, got {output_batched_models_multiple_data.shape}"
    console.print(f"[success]Output shape is correct: {expected_shape_ex3}")
    console.print(
        f"  Output for 1st model, 1st sample in its batch (first 5 logits):"
    )
    pprint(
        output_batched_models_multiple_data[0, 0, :5].detach().numpy(),
        console=console,
    )

    console.rule(
        "[bold green]Model forward pass demonstrations finished[/bold green]"
    )
