# ---INFO-----------------------------------------------------------------------
# Author(s):       Aditya Prakash
# Last Modified:   2023-10-05

# --Needed functionalities
# 1. Reimplement models based on standard base

# ---DEPENDENCIES---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt


# ---PLOTS----------------------------------------------------------------------
# Plots orginal and reconstructed images side by side, adaptive towards batch
# size, resizes everything to be as square as possible
def plot_reconstructions(xi, xd, title):
    num_images = xd.shape[0]
    num_rows = int(np.ceil(np.sqrt(num_images)))
    num_cols = int(np.ceil(num_images / num_rows))

    axes = plt.subplots(num_rows, 2 * num_cols, figsize=(2 * num_cols, num_rows))[1]
    axes = axes.flatten()

    for i in range(num_images):
        ax_pred = axes[2 * i]
        ax_true = axes[2 * i + 1]
        pred = np.array(xd[i, ...])
        true = np.array(xi[i, ...])
        ax_pred.imshow(pred)
        if i < num_cols:
            ax_pred.set_title("R")
        ax_pred.axis("off")
        ax_true.imshow(true)
        if i < num_cols:
            ax_true.set_title("O")
        ax_true.axis("off")
    for i in range(2 * num_images, len(axes)):
        axes[i].axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
