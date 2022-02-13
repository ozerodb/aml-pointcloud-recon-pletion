import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_model_summary import summary as model_summary

class bcolors:
    LIGHT_RED = "\033[0;31m"
    LIGHT_GREEN = "\033[0;32m"
    LIGHT_BLUE = "\033[0;34m"
    BOLD = "\033[1m"
    ENDC = '\033[0m'

def pretty_time_delta(seconds):
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%dd %dh %dm %ds' % (days, hours, minutes, seconds)
    elif hours > 0:
        return '%dh %dm %ds' % (hours, minutes, seconds)
    elif minutes > 0:
        return '%dm %ds' % (minutes, seconds)
    else:
        return '%ds' % (seconds,)

def plot_losses(train_losses, valid_losses):
    fig, ax = plt.subplots()
    ax.plot(range(len(train_losses)), train_losses, label='Train loss', marker='.', markersize=5, linewidth=0.8)
    ax.plot(range(len(valid_losses)), valid_losses, label='Valid loss', marker='.', markersize=5, linewidth=0.8)
    ax.set(xlabel='Epoch', ylabel='Loss')
    ax.axhline(y=min(train_losses), alpha=0.6, linestyle='--', color='cornflowerblue', linewidth=0.95)
    ax.axhline(y=min(valid_losses), alpha=0.6, linestyle='--', color='sandybrown', linewidth=0.95)
    ax.legend()
    ax.grid()
    plt.show()

def save_training_report(report_data, out_path):
    report_id = report_data['id']
    args = report_data['args']
    device = report_data['device']
    model = report_data['model']
    train_losses = report_data['train_losses']
    valid_losses = report_data['valid_losses']

    with open(out_path, 'w') as f:
        f.write(f"#### REPORT {report_id} ####\n")

        f.write(f"This training session can be replicated using the following arguments:\n")
        for key, value in vars(args).items():
            f.write(f"--{key}={value}\n")

        f.write('\n## MODEL ARCHITECTURE ##\n')
        f.write(model_summary(model, torch.zeros((args.batch_size, 1024, 3)).to(device)))

        f.write('\n## RESULTS ##\n')
        best_epoch=0
        for epoch in range(min(len(train_losses), len(valid_losses))):
            f.write(f'AFTER {epoch+1} EPOCHS, train loss: {train_losses[epoch]}, val loss: {valid_losses[epoch]}\n')
            best_epoch = epoch if valid_losses[epoch] < valid_losses[best_epoch] else best_epoch
        f.write(f'Best validation loss {valid_losses[best_epoch]} at epoch {best_epoch}\n')
    print(f"Saved report at {out_path}")

def pick_random_point_on_sphere():
    vec = np.random.randn(1, 3)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def pick_evenly_spaced_points_on_sphere(m=100):
    indices = np.arange(0, m, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices/m)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    ##todo