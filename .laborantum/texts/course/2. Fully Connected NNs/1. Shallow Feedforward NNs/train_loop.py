from html import escape

import torch
from tqdm.auto import tqdm

from .utils import mlflow as mlflow_utils


def _value_text(value):
    return '-' if value is None else f'{value:8.4f}'


def _number(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'item'):
        return value.item()
    return value


def _normalize_named_functions(functions, default_name):
    if isinstance(functions, dict):
        return functions
    return {default_name: functions}


def _loss_metric_name(stage, loss_name):
    if loss_name == 'loss':
        return f'{stage}_loss'
    return f'{stage}_loss/{loss_name}'


def _batch_status(batch_losses, batch_loss_emas):
    lines = ['batch losses:']
    if not batch_losses:
        lines.append('-')
    else:
        for loss_name, loss_value in batch_losses.items():
            loss_ema = None if batch_loss_emas is None else batch_loss_emas.get(loss_name)
            lines.append(
                f'{loss_name}:\t train {_value_text(loss_value)} | ema {_value_text(loss_ema)}'
            )
    return '\n'.join(lines)


def _set_bar_message(bar, message):
    children = getattr(getattr(bar, 'container', None), 'children', ())
    if len(children) >= 3:
        container = bar.container
        _, progress_widget, right_text = children

        container.layout.flex_flow = 'row wrap'
        container.layout.align_items = 'center'
        progress_widget.layout.flex = '1 1 auto'
        right_text.layout.width = '100%'
        right_text.layout.flex = '0 0 100%'
        right_text.layout.margin = '2px 0 0 0'

        bar.set_postfix_str('', refresh=True)
        progress_text = right_text.value
        right_text.value = (
            f'<div>{progress_text}</div>'
            '<pre style="margin:2px 0 0 0;font-family:monospace;white-space:pre;">'
            f'{escape(message)}'
            '</pre>'
        )
    else:
        bar.set_postfix_str(message.replace('\n', ' | '))


def _keep_bar_message_on_close(bar, get_message):
    original_close = bar.close

    def close_with_message():
        original_close()
        message = get_message()
        if message is not None:
            _set_bar_message(bar, message)

    bar.close = close_with_message


def _epoch_status(valid_losses, train_losses, valid_metrics=None, train_metrics=None):
    lines = [f'epoch loss:']

    loss_names = list((valid_losses or {}).keys())
    loss_names += [
        loss_name
        for loss_name in (train_losses or {})
        if loss_name not in loss_names
    ]
    if not loss_names:
        lines.append('-')
    else:
        for loss_name in loss_names:
            valid_loss = None if valid_losses is None else valid_losses.get(loss_name)
            train_loss = None if train_losses is None else train_losses.get(loss_name)
            lines.append(
                f'{loss_name}:\t val {_value_text(valid_loss)} | train {_value_text(train_loss)}'
            )

    lines.append(f'epoch metrics:')

    if not valid_metrics and not train_metrics:
        lines.append('-')
    else:
        metric_names = list((valid_metrics or {}).keys())
        metric_names += [
            metric_name
            for metric_name in (train_metrics or {})
            if metric_name not in metric_names
        ]
        for metric_name in metric_names:
            valid_metric = None if valid_metrics is None else valid_metrics.get(metric_name)
            train_metric = None if train_metrics is None else train_metrics.get(metric_name)
            lines.append(
                f'{metric_name}:\t val {_value_text(valid_metric)} | train {_value_text(train_metric)}'
            )

    return '\n'.join(lines)


def _empty_metric_totals(metrics):
    return {
        metric_name: {'enumerator': 0.0, 'denominator': 1.0e-8}
        for metric_name in metrics
    }


def _empty_loss_totals(losses):
    return {
        loss_name: {'enumerator': 0.0, 'denominator': 1.0e-8}
        for loss_name in losses
    }


def _compute_losses(losses, batch):
    return {
        ## YOUR CODE HERE
        loss_name: batch['signals']['output'].sum() * 0.0
        for loss_name in losses
    }


def _sum_losses(batch_losses):
    total_loss = None
    for loss_value in batch_losses.values():
        ...
        ## YOUR CODE HERE
        total_loss = loss_value if total_loss is None else total_loss
    return total_loss


def _update_loss_totals(loss_totals, batch_losses):
    for loss_name, loss_value in batch_losses.items():
        loss_totals[loss_name]['enumerator'] += _number(loss_value)
        loss_totals[loss_name]['denominator'] += 1.0


def _update_loss_emas(loss_emas, batch_losses, beta=0.90):
    for loss_name, loss_value in batch_losses.items():
        loss_number = _number(loss_value)
        if loss_emas[loss_name] is None:
            loss_emas[loss_name] = loss_number
        else:
            loss_emas[loss_name] = beta * loss_emas[loss_name] + (1 - beta) * loss_number


def _finalize_loss_totals(loss_totals):
    return {
        loss_name: totals['enumerator'] / totals['denominator']
        for loss_name, totals in loss_totals.items()
    }


def _update_metric_totals(metric_totals, metrics, batch):
    
    
    for metric_name, metric_fn in metrics.items():
        ...
        ## YOUR CODE HERE


def _finalize_metric_totals(metric_totals):
    return {
        metric_name: _number(totals['enumerator'] / totals['denominator'])
        for metric_name, totals in metric_totals.items()
    }


def train_model(
    model,
    n_epochs,
    train_dl,
    valid_dl,
    loss,
    optimizer,
    metrics=None,
    scheduler=None,
    mlflow_logger=None,
    run_name='model',
):
    losses = _normalize_named_functions(loss, 'loss')
    if not isinstance(losses, dict):
        raise TypeError('loss must be a function or a dictionary mapping loss names to loss functions')
    if not losses:
        raise ValueError('loss must contain at least one loss function')
    train_loss_history = {loss_name: [] for loss_name in losses}
    valid_loss_history = {loss_name: [] for loss_name in losses}
    metrics = {} if metrics is None else metrics
    if not isinstance(metrics, dict):
        raise TypeError('metrics must be a dictionary mapping metric names to metric functions')
    train_metrics_history = {metric_name: [] for metric_name in metrics}
    valid_metrics_history = {metric_name: [] for metric_name in metrics}
    batch_bar_total = len(train_dl)

    if mlflow_logger is not None:
        mlflow_logger.log_params(mlflow_utils.collect_training_params(
            run_name,
            model,
            loss,
            optimizer,
            scheduler,
            getattr(train_dl, 'batch_size', None),
            n_epochs,
            train_dl,
            valid_dl,
        ))

    epoch_width = len(str(n_epochs))
    final_epoch_message = None
    final_batch_message = None

    with tqdm(total=n_epochs, desc='epochs', position=0) as epoch_bar:
        _keep_bar_message_on_close(epoch_bar, lambda: final_epoch_message)
        with tqdm(total=batch_bar_total, desc='batches', position=1, leave=True) as batch_bar:
            _keep_bar_message_on_close(batch_bar, lambda: final_batch_message)
            for epoch in range(n_epochs):
                train_losses = _empty_loss_totals(losses)
                valid_losses = _empty_loss_totals(losses)
                loss_emas = {loss_name: None for loss_name in losses}

                train_metrics = _empty_metric_totals(metrics)
                valid_metrics = _empty_metric_totals(metrics)

                batch_bar.reset(total=batch_bar_total)
                batch_bar.set_description_str(f'epoch {epoch + 1:>{epoch_width}}/{n_epochs}')
                final_batch_message = _batch_status(None, None)
                _set_bar_message(batch_bar, final_batch_message)

                for batch_index, batch in enumerate(train_dl):
                    batch = {'data': batch}

                    ## YOUR CODE HERE
                    # Implement one training step:
                    # switch to training mode
                    # reset gradients
                    # run the model
                    # compute named losses
                    # store named losses and their sum
                    # backpropagate through the summed loss
                    # update weights,
                    # switch the model to evaluation mode
                    # update metric numerators/denominators.
                    model.train()
                    optimizer.zero_grad()
                    model(batch)
                    loss_values = _compute_losses(losses, batch)
                    loss_value = _sum_losses(loss_values)
                    batch['losses'] = loss_values
                    batch['loss'] = loss_value
                    loss_value.backward()
                    optimizer.step()
                    model.eval()

                    _update_loss_totals(train_losses, loss_values)
                    _update_loss_emas(loss_emas, loss_values)

                    batch_bar.update(1)

                    final_batch_message = _batch_status(loss_values, loss_emas)
                    _set_bar_message(batch_bar, final_batch_message)

                    if mlflow_logger is not None:
                        global_step = epoch * len(train_dl) + batch_index
                        for loss_name, loss_value in loss_values.items():
                            mlflow_logger.log_metric(
                                _loss_metric_name('batch/train', loss_name),
                                loss_value,
                                step=global_step,
                            )

                with torch.no_grad():
                    for valid_batch in valid_dl:
                        valid_batch = {'data': valid_batch}

                        ## YOUR CODE HERE
                        # Implement one validation step:
                        # switch the model to evaluation mode
                        # run the model without gradients
                        # compute and store named validation losses
                        # and update metric numerators/denominators. 
                        # Do not call backward() or step().
                        model.eval()
                        model(valid_batch)
                        valid_loss_values = _compute_losses(losses, valid_batch)
                        valid_batch['losses'] = valid_loss_values
                        valid_batch['loss'] = _sum_losses(valid_loss_values)

                        _update_loss_totals(valid_losses, valid_loss_values)

                finalized_train_losses = _finalize_loss_totals(train_losses)
                finalized_valid_losses = _finalize_loss_totals(valid_losses)

                for loss_name in losses:
                    train_loss_history[loss_name].append(finalized_train_losses[loss_name])
                    valid_loss_history[loss_name].append(finalized_valid_losses[loss_name])

                finalized_train_metrics = _finalize_metric_totals(train_metrics)
                finalized_valid_metrics = _finalize_metric_totals(valid_metrics)
                for metric_name in metrics:
                    train_metrics_history[metric_name].append(finalized_train_metrics[metric_name])
                    valid_metrics_history[metric_name].append(finalized_valid_metrics[metric_name])

                if mlflow_logger is not None:
                    for loss_name in losses:
                        mlflow_logger.log_metric(
                            _loss_metric_name('epoch/train', loss_name),
                            finalized_train_losses[loss_name],
                            step=epoch,
                        )
                        mlflow_logger.log_metric(
                            _loss_metric_name('epoch/valid', loss_name),
                            finalized_valid_losses[loss_name],
                            step=epoch,
                        )

                    for metric_name in metrics:
                        mlflow_logger.log_metric(
                            f'epoch/train_{metric_name}',
                            finalized_train_metrics[metric_name],
                            step=epoch,
                        )
                        mlflow_logger.log_metric(
                            f'epoch/valid_{metric_name}',
                            finalized_valid_metrics[metric_name],
                            step=epoch,
                        )

                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(sum(finalized_valid_losses.values()))
                    else:
                        scheduler.step()

                epoch_bar.update(1)
                final_epoch_message = _epoch_status(
                    finalized_valid_losses,
                    finalized_train_losses,
                    finalized_valid_metrics,
                    finalized_train_metrics,
                )
                _set_bar_message(
                    epoch_bar,
                    final_epoch_message,
                )

    if final_epoch_message is not None:
        _set_bar_message(epoch_bar, final_epoch_message)
    if final_batch_message is not None:
        _set_bar_message(batch_bar, final_batch_message)

    history = {
        'train_loss': train_loss_history,
        'valid_loss': valid_loss_history,
        'train_metrics': train_metrics_history,
        'valid_metrics': valid_metrics_history,
    }

    if mlflow_logger is not None:
        mlflow_utils.plot_training_curves(mlflow_logger, history, run_name)

    return history
