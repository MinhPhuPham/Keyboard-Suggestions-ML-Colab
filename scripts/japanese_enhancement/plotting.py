"""
Plotting utilities for Multi-Task GRU training.
"""
import matplotlib.pyplot as plt

from . import config


def plot_training_history(history):
    """Plot training curves for both KKC and NWP heads.

    Creates a 2×2 grid:
    - Top-left: KKC loss
    - Top-right: NWP loss
    - Bottom-left: KKC accuracy
    - Bottom-right: NWP accuracy
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Multi-Task GRU Training', fontsize=14, fontweight='bold')

    # KKC Loss
    ax = axes[0, 0]
    ax.plot(history['kkc_loss'], label='Train')
    ax.plot(history['val_kkc_loss'], label='Val')
    ax.set_title('KKC Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # NWP Loss
    ax = axes[0, 1]
    ax.plot(history['nwp_loss'], label='Train')
    ax.plot(history['val_nwp_loss'], label='Val')
    ax.set_title('NWP Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # KKC Accuracy
    ax = axes[1, 0]
    ax.plot([x*100 for x in history['kkc_accuracy']], label='Train')
    ax.plot([x*100 for x in history['val_kkc_accuracy']], label='Val')
    ax.set_title('KKC Accuracy (%)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # NWP Accuracy
    ax = axes[1, 1]
    ax.plot([x*100 for x in history['nwp_accuracy']], label='Train')
    ax.plot([x*100 for x in history['val_nwp_accuracy']], label='Val')
    ax.set_title('NWP Accuracy (%)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{config.MODEL_DIR}/training_multitask.png', dpi=150)
    plt.show()

    # Print summary
    losses = history['loss']
    print(f"\n📊 Training Summary:")
    print(f"  Total loss: {losses[0]:.4f} → {losses[-1]:.4f} "
          f"({'✅ decreasing' if losses[-1] < losses[0] else '❌ NOT decreasing'})")
    print(f"  KKC acc:    {history['kkc_accuracy'][0]*100:.1f}% → "
          f"{history['kkc_accuracy'][-1]*100:.1f}%")
    print(f"  NWP acc:    {history['nwp_accuracy'][0]*100:.1f}% → "
          f"{history['nwp_accuracy'][-1]*100:.1f}%")
    print(f"  Best val KKC acc: {max(history['val_kkc_accuracy'])*100:.1f}%")
    print(f"  Best val NWP acc: {max(history['val_nwp_accuracy'])*100:.1f}%")
