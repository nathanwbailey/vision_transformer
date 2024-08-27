import torch

from model_building_blocks import (CreatePatchesLayer, PatchEmbeddingLayer,
                                   TransformerBlock, create_mlp_block)


class ViTClassifierModel(torch.nn.Module):
    """ViT Model for Image Classification."""

    def __init__(
        self,
        num_transformer_layers: int,
        embed_dim: int,
        feed_forward_dim: int,
        num_heads: int,
        patch_size: int,
        num_patches: int,
        mlp_head_units: list[int],
        num_classes: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        """Init Function."""
        super().__init__()
        self.create_patch_layer = CreatePatchesLayer(patch_size, patch_size)
        self.patch_embedding_layer = PatchEmbeddingLayer(
            num_patches, batch_size, patch_size, embed_dim, device
        )
        self.transformer_layers = torch.nn.ModuleList()
        for _ in range(num_transformer_layers):
            self.transformer_layers.append(
                TransformerBlock(
                    num_heads, embed_dim, embed_dim, feed_forward_dim
                )
            )

        self.mlp_block = create_mlp_block(
            input_features=embed_dim,
            output_features=mlp_head_units,
            activation_function=torch.nn.GELU,
            dropout_rate=0.5,
        )

        self.logits_layer = torch.nn.Linear(mlp_head_units[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        x = self.create_patch_layer(x)
        x = self.patch_embedding_layer(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        x = x[:, 0]
        x = self.mlp_block(x)
        x = self.logits_layer(x)
        return x
