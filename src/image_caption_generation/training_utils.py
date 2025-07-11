import os
import torch
import matplotlib.pyplot as plt


def save_model_checkpoint(model, optimizer, scheduler, epoch, avg_epoch_loss, val_loss, metrics, model_save_dir, vocab_size, decoder_type, dataset, best_loss=None, is_best=False):
    """Save model and training state after each epoch. Optionally save best model."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'training_loss': avg_epoch_loss,
        'validation_loss': val_loss,
        'metrics': metrics,
        'vocab': dataset.vocab,
        'hyperparameters': {
            'embed_size': getattr(model.encoder, 'embed', None).out_features if hasattr(model.encoder, 'embed') else None,
            'hidden_size': model.decoder.hidden_size if hasattr(model.decoder, 'hidden_size') else None,
            'vocab_size': vocab_size,
            'num_layers': model.decoder.num_layers,
            'decoder_type': decoder_type
        }
    }
    filename = os.path.join(model_save_dir, f"model_epoch{epoch}.pth")
    torch.save(checkpoint, filename)
    if is_best:
        best_filename = os.path.join(model_save_dir, "best_model.pth")
        torch.save(checkpoint, best_filename)


def save_loss_curves(epochs_list, training_losses, validation_losses, model_save_dir):
    plt.figure(figsize=(8,4))
    plt.plot(epochs_list, training_losses, label='Train Loss')
    plt.plot(epochs_list, validation_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training/Validation Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, 'loss_curve.png'))
    plt.close()


def save_metrics_history(metrics_history, model_save_dir):
    import json
    metrics_path = os.path.join(model_save_dir, 'metrics_history.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)
