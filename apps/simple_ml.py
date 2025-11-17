"""hw1/apps/simple_ml.py"""

import copy
import struct
import gzip
import math
import time
import numpy as np

from tqdm.auto import tqdm
import apps.config_utils as config_utils

import sys

sys.path.append("python/")
import needle as ndl
import needle.nn as nn


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname, "rb") as _f:
      image_data = _f.read()
      dim_of_data = 28*28
      X = np.frombuffer(image_data, np.uint8, offset=16).reshape(-1, dim_of_data).astype('float32') / 255.0
    
    with gzip.open(label_filename, "rb") as _f:
      label_data = _f.read()
      y = np.frombuffer(label_data, np.uint8, offset=8)
    
    return (X,y)
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION

    log_sum_exp = ndl.log(ndl.exp(Z).sum((1,))).sum()
    z_y = (y_one_hot * Z).sum()
    return (log_sum_exp - z_y) / Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    for start in range(0, num_examples, batch):
      end = min(start+batch, num_examples)
      X_batch = ndl.Tensor(X[start:end])
      y_batch = y[start:end]
      batch_size = X_batch.shape[0]
      y_one_hot = np.zeros((batch, y.max() +1))
      y_one_hot[np.arange(batch), y_batch] = 1
      y_one_hot = ndl.Tensor(y_one_hot)

      Z = ndl.relu(X_batch.matmul(W1)).matmul(W2)
      loss = softmax_loss(Z, y_one_hot)
      loss.backward()

      # # Similar to what was done in the stochastic grad for the results
      W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
      W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
    return W1, W2
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(
    dataloader,
    model,
    loss_fn=nn.SoftmaxLoss(),
    opt=None,
    *,
    progress_bar=False,
    epoch_index=None,
    total_epochs=None,
    max_batches=None,
):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if dataloader is None:
        raise ValueError("dataloader cannot be None")
    
    if opt is not None:
        model.train()
    else:
        model.eval()
    
    correct = 0
    total_loss = 0.0
    total_samples = 0
    
    # Setup progress bar if requested
    pbar = None
    use_progress = progress_bar and dataloader is not None
    if use_progress:
        batch_size = getattr(dataloader, "batch_size", None) or 1
        dataset = getattr(dataloader, "dataset", None)
        dataset_len = len(dataset) if dataset is not None else None
        total_batches = None
        if dataset_len is not None and batch_size:
            total_batches = math.ceil(dataset_len / batch_size)
        if max_batches is not None:
            if total_batches is not None:
                total_batches = min(total_batches, max_batches)
            else:
                total_batches = max_batches
        
        if epoch_index is not None and total_epochs is not None:
            desc = f"Epoch {epoch_index}/{total_epochs}"
        elif epoch_index is not None:
            desc = f"Epoch {epoch_index}"
        else:
            desc = "Epoch"
        
        # Create progress bar (standalone, not wrapping dataloader)
        if total_batches is not None:
            pbar = tqdm(total=int(total_batches), desc=desc, leave=False)
        else:
            pbar = tqdm(desc=desc, leave=False)
    
    batch_counter = 0
    try:
        for batch in dataloader:
            if max_batches is not None and batch_counter >= max_batches:
                break
            
            if pbar is not None:
                pbar.update(1)
            
            batch_counter += 1
            X, y = batch
            
            if opt is not None:
                opt.reset_grad()
            
            out = model(X)
            
            loss = loss_fn(out, y)
            
            # Compute accuracy
            predictions = np.argmax(out.numpy(), axis=1)
            y_np = y.numpy()
            if len(y_np.shape) > 1:
                y_np = y_np.flatten()
            batch_correct = np.sum(predictions == y_np)
            correct += batch_correct
            total_samples += y_np.shape[0]
            
            # Accumulate loss (weighted by batch size)
            loss_val = loss.data.numpy()
            # Convert numpy array/scalar to Python float
            if isinstance(loss_val, np.ndarray):
                loss_val = float(loss_val.item() if loss_val.size == 1 else loss_val)
            else:
                # Handle numpy scalars (np.float32, etc.)
                loss_val = float(loss_val)
            total_loss += loss_val * float(y_np.shape[0])
            
            if opt is not None:
                loss.backward()
                opt.step()
    finally:
        if use_progress and pbar is not None:
            pbar.close()
    
    # Ensure all values are Python native types, not numpy scalars
    # Use .item() for numpy scalars, float() for everything else
    if isinstance(correct, (np.integer, np.floating)):
        correct = float(correct.item())
    else:
        correct = float(correct)
    
    if isinstance(total_samples, (np.integer, np.floating)):
        total_samples = float(total_samples.item())
    else:
        total_samples = float(total_samples)
    
    if isinstance(total_loss, (np.integer, np.floating)):
        total_loss = float(total_loss.item())
    else:
        total_loss = float(total_loss)
    
    avg_acc = float(correct / total_samples) if total_samples > 0 else 0.0
    avg_loss = float(total_loss / total_samples) if total_samples > 0 else 0.0
    
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def _instantiate_loss(loss_fn, loss_config):
    """Instantiate a loss module from configuration or fallback value."""
    if loss_config is None:
        return loss_fn() if isinstance(loss_fn, type) else loss_fn

    if isinstance(loss_config, nn.Module):
        return loss_config

    if isinstance(loss_config, str):
        if not hasattr(nn, loss_config):
            raise AttributeError(f"needle.nn has no loss named '{loss_config}'.")
        loss_cls = getattr(nn, loss_config)
        return loss_cls()

    if isinstance(loss_config, dict):
        loss_cfg = dict(loss_config)
        if "callable" in loss_cfg:
            factory = loss_cfg.pop("callable")
            if callable(factory):
                return factory(**loss_cfg)
            raise TypeError("'callable' entry in loss config must be callable.")
        name = loss_cfg.pop("name", None)
        if name is None:
            raise KeyError("Loss config must include a 'name' when provided as a dict.")
        if not hasattr(nn, name):
            raise AttributeError(f"needle.nn has no loss named '{name}'.")
        loss_cls = getattr(nn, name)
        return loss_cls(**loss_cfg)

    if callable(loss_config):
        return loss_config()

    raise TypeError("Unsupported loss configuration type.")


def _resolve_optimizer(model, optimizer, lr, weight_decay, config):
    """Return an optimizer instance based on explicit args or configuration.
    
    Priority order:
    1. If config["optimizer"] exists, use it (config takes precedence)
    2. If optimizer is an Optimizer instance, use it
    3. If optimizer is a callable (class), instantiate it with lr and weight_decay
    4. Otherwise raise TypeError
    """
    # Config takes highest priority - if optimizer is specified in config, use it
    optimizer_cfg = None if config is None else config.get("optimizer")
    if optimizer_cfg is not None:
        return ndl.optim.build_optimizer_from_config(model.parameters(), optimizer_cfg)

    # If no config optimizer, fall back to explicit optimizer parameter
    if optimizer is None:
        raise ValueError(
            "optimizer must be provided either as a parameter or in config['optimizer']. "
            "If using config, set config['optimizer'] = {'name': 'adam', ...}"
        )
    
    if isinstance(optimizer, ndl.optim.Optimizer):
        return optimizer

    if not callable(optimizer):
        raise TypeError("optimizer must be an Optimizer instance or class when config is not provided.")

    return optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
        lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss, config=None, metrics_callback=None):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class or instance. Ignored when config["optimizer"] is provided.
        lr: learning rate (float). Overrides default when not set in config.
        weight_decay: weight decay (float). Overrides default when not set in config.
        loss_fn: nn.Module class or instance
        config: Optional dictionary describing run/optimizer/loss configuration.
        metrics_callback: Optional callable receiving per-epoch metric dictionaries.

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training.
        avg_loss: average loss over dataset from last epoch of training.
        history (optional): when config["return_history"] is True, a list of per-epoch metric dicts.
    """
    config = copy.deepcopy(config) if config is not None else {}

    run_config = config.get("run", {})
    if run_config and not isinstance(run_config, dict):
        raise TypeError("'run' section of config must be a dictionary if provided.")

    seed = run_config.get("seed", 4)
    np.random.seed(seed)
    epochs_override = run_config.get("epochs")
    if epochs_override is not None:
        n_epochs = epochs_override

    if n_epochs <= 0:
        raise ValueError("Number of epochs must be positive.")

    record_history = config.get("return_history", False)
    history = []

    opt = _resolve_optimizer(model, optimizer, lr, weight_decay, config)
    loss_cfg = config.get("loss")
    loss_fn_instance = _instantiate_loss(loss_fn, loss_cfg)
    progress_enabled = config.get("progress_bar", False)
    max_batches = run_config.get("max_batches")
    verbose = run_config.get("verbose", True)

    for epoch in range(n_epochs):
        start_time = time.time()
        if verbose:
            info_prefix = f"[Epoch {epoch + 1}/{n_epochs}]"
            if max_batches:
                print(f"{info_prefix} starting ({max_batches} batches cap)...")
            else:
                print(f"{info_prefix} starting...")
        train_acc, train_loss = epoch_general_cifar10(
            dataloader,
            model,
            loss_fn=loss_fn_instance,
            opt=opt,
            progress_bar=progress_enabled,
            epoch_index=epoch + 1,
            total_epochs=n_epochs,
            max_batches=max_batches,
        )
        # Ensure metrics are Python floats for downstream consumers / logging
        if isinstance(train_acc, (np.ndarray, np.integer, np.floating)):
            train_acc = float(np.asarray(train_acc).item())
        else:
            train_acc = float(train_acc)
        if isinstance(train_loss, (np.ndarray, np.integer, np.floating)):
            train_loss = float(np.asarray(train_loss).item())
        else:
            train_loss = float(train_loss)
        elapsed = time.time() - start_time
        if verbose:
            print(f"[Epoch {epoch + 1}/{n_epochs}] finished in {elapsed:.2f}s")
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "learning_rate": getattr(opt, "lr", lr),
            "duration_sec": elapsed,
            "max_batches": max_batches,
        }
        history.append(epoch_metrics)
        if callable(metrics_callback):
            metrics_callback(epoch_metrics)

    avg_acc = history[-1]["train_acc"] if history else 0.0
    avg_loss = history[-1]["train_loss"] if history else 0.0

    if record_history:
        return avg_acc, avg_loss, history

    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def run_cifar10_grid_search(
    model_builder,
    dataloader_builder,
    base_config,
    sweep_spec,
    *,
    loss_fn=nn.SoftmaxLoss,
    optimizer=ndl.optim.Adam,
):
    """Run a grid search over optimizer and training hyperparameters for CIFAR-10.

    Args:
        model_builder: Callable returning a freshly initialized model.
        dataloader_builder: Callable returning a new dataloader per run.
        base_config: Base configuration dictionary shared across runs.
        sweep_spec: Mapping of dotted paths to sequences of candidate values.
        loss_fn: Optional loss module/class override.
        optimizer: Optional optimizer class override when config lacks one.

    Returns:
        List of dictionaries containing run metadata and results.
    """
    results = []
    for run_id, cfg in enumerate(config_utils.grid_sweep(base_config, sweep_spec), start=1):
        run_config = config_utils.merge_run_config(cfg, {"return_history": True})
        model = model_builder()
        dataloader = dataloader_builder()

        output = train_cifar10(
            model,
            dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=run_config,
        )

        if len(output) == 3:
            final_acc, final_loss, history = output
        else:
            final_acc, final_loss = output
            history = []

        results.append(
            {
                "run_id": run_id,
                "config": run_config,
                "final_acc": final_acc,
                "final_loss": final_loss,
                "history": history,
            }
        )

    return results


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_fn_instance = loss_fn()
    avg_acc, avg_loss = epoch_general_cifar10(
        dataloader,
        model,
        loss_fn=loss_fn_instance,
        opt=None,
        progress_bar=False,
        max_batches=None,
    )
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def split_cifar10_train_test(dataset, train_ratio=0.8, seed=42):
    """
    Split a CIFAR-10 dataset into train and test subsets.
    
    Args:
        dataset: CIFAR10Dataset instance
        train_ratio: Ratio of data to use for training (default 0.8 for 80%)
        seed: Random seed for reproducibility
        
    Returns:
        train_dataset: Subset for training
        test_dataset: Subset for testing
    """
    np.random.seed(seed)
    total_size = len(dataset)
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    
    train_size = int(total_size * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Create subset datasets
    train_dataset = ndl.data.DatasetSubset(dataset, train_indices)
    test_dataset = ndl.data.DatasetSubset(dataset, test_indices)
    
    return train_dataset, test_dataset


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss, opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()
    
    # Instantiate loss_fn if it's a class
    if isinstance(loss_fn, type):
        loss_fn_instance = loss_fn()
    else:
        loss_fn_instance = loss_fn
    
    total_loss = 0.0
    total_correct = 0
    cnt = 0
    nbatch = data.shape[0]
    hidden = None
    
    i = 0
    while i < nbatch - 1:
        x, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        batch_tokens = y.shape[0]
        if batch_tokens == 0:
            break
        
        if hidden is not None:
            if isinstance(hidden, tuple):
                hidden = tuple(h.detach() for h in hidden)
            else:
                hidden = hidden.detach()
        
        if opt is not None:
            opt.reset_grad()
        
        y_pred, hidden = model(x, hidden)
        loss = loss_fn_instance(y_pred, y)
        
        if opt is not None:
            loss.backward()
            if clip is not None:
                total_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad = param.grad.numpy()
                        total_norm += np.sum(grad * grad)
                total_norm = np.sqrt(total_norm)
                if total_norm > clip:
                    coef = clip / (total_norm + 1e-6)
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad = param.grad * coef
            opt.step()
        
        cnt += batch_tokens
        total_loss += loss.numpy().item() * batch_tokens
        total_correct += np.sum(y_pred.numpy().argmax(axis=1) == y.numpy())
        i += seq_len
    
    avg_acc = total_correct / cnt
    avg_loss = total_loss / cnt
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
        lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
        device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(
            data=data,
            model=model,
            seq_len=seq_len,
            loss_fn=loss_fn,
            opt=opt,
            clip=clip,
            device=device,
            dtype=dtype
        )
    
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(
        data=data,
        model=model,
        seq_len=seq_len,
        loss_fn=loss_fn,
        opt=None,
        clip=None,
        device=device,
        dtype=dtype
    )
    
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
