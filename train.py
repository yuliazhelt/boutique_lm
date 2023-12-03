import torch
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
# from IPython.display import clear_output
from tqdm import tqdm
from model import LanguageModel
import wandb

plt.rcParams.update({'font.size': 15})


def plot_losses(train_losses: List[float], val_losses: List[float]):
    """
    Plot loss and perplexity of train and validation samples
    :param train_losses: list of train losses at each epoch
    :param val_losses: list of validation losses at each epoch
    """
    # clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')

    """
    Calculate train and validation perplexities given lists of losses
    """
    train_perplexities, val_perplexities = np.exp(train_losses), np.exp(val_losses)

    axs[1].plot(range(1, len(train_perplexities) + 1), train_perplexities, label='train')
    axs[1].plot(range(1, len(val_perplexities) + 1), val_perplexities, label='val')
    axs[1].set_ylabel('perplexity')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()


def training_epoch(model: LanguageModel, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, val_loader: DataLoader, grad_accumulate_period: int, val_log_period: int, tqdm_desc: str):
    """
    Process one training epoch
    :param model: language model to train
    :param optimizer: optimizer instance
    :param criterion: loss function class
    :param loader: training dataloader
    :param tqdm_desc: progress bar description
    :param grad_accumulate_period: steps for gradient accumulation
    :return: running train loss
    """
    train_loss = 0.0

    model.train()
    step = 0
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        """
        Process one training step: calculate loss,
        call backward and make one optimizer step.
        Accumulate sum of losses for different batches in train_loss
        """
        indices = indices[:, :lengths.max()].to(model.device)
        logits = model(indices[:, :-1])
        loss = criterion(logits.transpose(1, 2), indices[:, 1:])
        loss.backward()
        if step % grad_accumulate_period == 0:
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({"train_loss": loss.item(), "train_perplexity": np.exp(loss.item())})

        train_loss += loss.item() * indices.shape[0]


        if step > 0 and step % val_log_period == 0:
            validation_epoch(
                model, criterion, val_loader, tqdm_desc=""
            )
            model.train()
        step += 1

        
        
    train_loss /= len(loader.dataset)
    return train_loss


@torch.no_grad()
def validation_epoch(model: LanguageModel, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):
    """
    Process one validation epoch
    :param model: language model to validate
    :param criterion: loss function class
    :param loader: validation dataloader
    :param tqdm_desc: progress bar description
    :return: validation loss
    """
    val_loss = 0.0

    model.eval()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        """
        Process one validation step: calculate loss.
        Accumulate sum of losses for different batches in val_loss
        """
        indices = indices[:, :lengths.max()].to(model.device)
        logits = model(indices[:, :-1])
        loss = criterion(logits.transpose(1, 2), indices[:, 1:])
        val_loss += loss.item() * indices.shape[0]

    val_loss /= len(loader.dataset)
    wandb.log({"val_loss": val_loss, "val_perplexity": np.exp(val_loss)})

    return val_loss


def train(model: LanguageModel, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, grad_accumulate_period: int = 1, num_examples=5):
    """
    Train language model for several epochs
    :param model: language model to train
    :param optimizer: optimizer instance
    :param scheduler: optional scheduler
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param num_epochs: number of training epochs
    :param grad_accumulate_period: steps for gradient accumulation
    :param num_examples: number of generation examples to print after each epoch
    """
    # wandb.init(project='boutique_lm')

    train_losses, val_losses = [], []
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_id)

    for epoch in range(2, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader, val_loader,
            grad_accumulate_period=grad_accumulate_period,
            val_log_period=6_000,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        val_losses += [val_loss]
        plot_losses(train_losses, val_losses)

        print('Generation examples:')
        sample_text_table = wandb.Table(columns=["id", "prompt", "temperature", "text"])
        for i in range(num_examples):
            prefix=""
            temp=np.random.uniform(0.1, 10)
            text_generated = model.inference(prefix=prefix, temp=temp)
            print(text_generated)
            sample_text_table.add_data(i, prefix, temp, text_generated)
        wandb.log({"text_samples" : sample_text_table})
