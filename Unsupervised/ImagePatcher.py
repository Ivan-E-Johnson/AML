import itk
import torch
import numpy as np
import torch.nn as nn
from monai.networks.blocks import PatchEmbeddingBlock
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt


class ImagePatcher(nn.Module):
    def __init__(
        self, in_channels, img_size, patch_size, hidden_size, num_heads, device="cpu"
    ):
        super(ImagePatcher, self).__init__()
        self.device = device
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        # Ensure that the patch size and image size are compatible
        self.patch_embed_block = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type="conv",  # 'conv' or 'perceptron'
            pos_embed_type="learnable",  # 'none', 'learnable', or 'sincos'
        ).to(device)

    def forward(self, x, mask_rate=0.75):
        # Apply patch embeddings
        embeddings = self.patch_embed_block(x)
        masked_embeddings, mask = self.apply_mask(embeddings, mask_rate)
        return masked_embeddings, mask

    def apply_mask(self, embeddings, mask_rate):
        num_patches = embeddings.shape[1]
        mask_indices = torch.randperm(num_patches)[: int(num_patches * mask_rate)]
        mask = torch.ones(num_patches, device=self.device)
        mask[mask_indices] = 0
        mask = mask.view(1, num_patches, 1).expand_as(embeddings)
        masked_embeddings = embeddings * mask
        return masked_embeddings, mask

    def visualize_patches(self, raw_image_tensor, embeddings, mask):
        # Calculate the number of rows and columns based on image and patch dimensions
        patch_height, patch_width = self.patch_size[1], self.patch_size[2]
        n_cols = raw_image_tensor.shape[4] // patch_width
        n_rows = raw_image_tensor.shape[3] // patch_height

        fig, axes = plt.subplots(
            figsize=(n_cols * 4, n_rows * 2), nrows=n_rows, ncols=n_cols * 2
        )

        # Ensure the unfolding is compatible with a 5D tensor (N, C, D, H, W)
        patches = raw_image_tensor.unfold(3, patch_height, patch_height).unfold(
            4, patch_width, patch_width
        )
        patches = patches.contiguous().view(-1, 1, patch_height, patch_width)

        for n in range(n_rows * n_cols):
            if n >= embeddings.shape[1]:
                axes[n // n_cols, 2 * (n % n_cols)].axis("off")
                axes[n // n_cols, 2 * (n % n_cols) + 1].axis("off")
                continue

            # Visualize original patch
            orig_patch_ax = axes[n // n_cols, 2 * (n % n_cols)]
            patch_img = patches[n].cpu()
            patch_img = ToPILImage()(patch_img)
            orig_patch_ax.imshow(patch_img)
            orig_patch_ax.set_title(f"Original Patch {n + 1}", fontsize=8)
            orig_patch_ax.axis("off")

            # Visualize embedding
            emb_patch_ax = axes[n // n_cols, 2 * (n % n_cols) + 1]
            patch = embeddings[0, n]  # First batch, nth patch
            if mask[0, n, 0] == 0:
                emb_patch_ax.imshow(
                    np.zeros((1, self.hidden_size)), cmap="hot", interpolation="nearest"
                )
                emb_patch_ax.set_title(f"Masked Embedding {n}", fontsize=8)
            else:
                side_length = int(np.sqrt(self.hidden_size))
                if side_length**2 == self.hidden_size:
                    reshaped_patch = (
                        patch.view(side_length, side_length).detach().numpy()
                    )
                else:
                    reshaped_patch = patch.view(1, self.hidden_size).detach().numpy()

                emb_patch_ax.imshow(reshaped_patch, cmap="hot", interpolation="nearest")
                emb_patch_ax.set_title(f"Embedding {n + 1}", fontsize=8)
            emb_patch_ax.axis("off")
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fn = "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/DATA/ALL_PROSTATEx/WITH_SEGMENTATION/PreProcessed/ProstateX-0004/ProstateX-0004_pp_t2w.nii.gz"  # Specify your file path here
    itk_image = itk.imread(fn)
    img_array = itk.GetArrayFromImage(itk_image)  # Z, Y, X
    img_tensor = (
        torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0).to(device)
    )  # N, C, D, H, W

    print(f"Image shape: {img_array.shape}")
    print(f"Image tensor shape: {img_tensor.shape}")
    patcher = ImagePatcher(
        in_channels=1,
        img_size=(32, 320, 320),
        patch_size=(32, 32, 32),
        hidden_size=1024,
        num_heads=8,
        device=device,
    )

    embeddings, mask = patcher(img_tensor)
    n_patches_per_dim = img_tensor.shape[-1] // patcher.patch_size[0]
    n_rows, n_cols = n_patches_per_dim, n_patches_per_dim
    patcher.visualize_patches(img_tensor, embeddings, mask)


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     patcher = ImagePatcher(in_channels=3, img_size=(64, 64, 64), patch_size=(16, 16, 16), hidden_size=256, num_heads=8, device=device)
#     raw_image_tensor = torch.randn(1, 3, 64, 64, 64).to(device)
#     embeddings, mask = patcher(raw_image_tensor)
#     n_patches_per_dim = raw_image_tensor.shape[-1] // patcher.patch_size[0]  # Assuming equal dimensions for simplicity
#     n_rows, n_cols = n_patches_per_dim, n_patches_per_dim
#     patcher.visualize_patches(embeddings, mask, n_rows=n_rows, n_cols=n_cols)
