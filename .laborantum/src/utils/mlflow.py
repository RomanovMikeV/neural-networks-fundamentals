from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
import subprocess
import sys
import tempfile
import time

import matplotlib.pyplot as plt
import torch

try:
    from IPython.display import Markdown, display
except ImportError:
    Markdown = None
    display = None

try:
    import mlflow as _mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    _mlflow = None
    MlflowClient = None
    MLFLOW_AVAILABLE = False


MLFLOW_EXPERIMENT_NAME = 'Shallow Feedforward NNs'
MLFLOW_UI_BASE_URL = 'http://127.0.0.1:5000'
MLFLOW_TRACKING_DIR = Path(__file__).resolve().parents[2] / 'mlflow_logs'
_MLFLOW_UI_PROCESS = None


def _display_markdown(text):
    if display is not None and Markdown is not None:
        display(Markdown(text))
    else:
        print(text)


def _is_mlflow_ui_running(ui_base_url, timeout=0.3):
    try:
        with urlopen(ui_base_url, timeout=timeout):
            return True
    except Exception:
        return False


def ensure_mlflow_ui(
        tracking_dir=MLFLOW_TRACKING_DIR,
        ui_base_url=MLFLOW_UI_BASE_URL,
        startup_timeout=8.0):
    global _MLFLOW_UI_PROCESS

    if not MLFLOW_AVAILABLE:
        return False

    if _is_mlflow_ui_running(ui_base_url):
        return True

    parsed_url = urlparse(ui_base_url)
    port = str(parsed_url.port or 5000)
    tracking_dir = Path(tracking_dir).resolve()

    if _MLFLOW_UI_PROCESS is None or _MLFLOW_UI_PROCESS.poll() is not None:
        command = [
            sys.executable,
            '-m',
            'mlflow',
            'ui',
            '--backend-store-uri',
            tracking_dir.as_uri(),
            '--host',
            '0.0.0.0',
            '--port',
            port,
            '--workers',
            '1',
            '--allowed-hosts',
            '*',
            '--cors-allowed-origins',
            '*',
        ]
        try:
            _MLFLOW_UI_PROCESS = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            return False

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if _is_mlflow_ui_running(ui_base_url):
            return True
        if _MLFLOW_UI_PROCESS.poll() is not None:
            return False
        time.sleep(0.25)

    if _is_mlflow_ui_running(ui_base_url, timeout=1.0):
        return True

    return _MLFLOW_UI_PROCESS.poll() is None


class NoOpMLflowLogger:
    enabled = False
    run_id = None

    def log_params(self, params):
        pass

    def log_metric(self, name, value, step=None):
        pass

    def log_figure(self, figure, artifact_path):
        pass

    def end(self):
        pass


class MLflowRunLogger:
    enabled = True

    def __init__(self, run, ui_base_url=MLFLOW_UI_BASE_URL):
        self.run = run
        self.run_id = run.info.run_id
        self.experiment_id = run.info.experiment_id
        self.ui_base_url = ui_base_url
        self.client = MlflowClient()

    @property
    def url(self):
        return f'{self.ui_base_url}/#/experiments/{self.experiment_id}/runs/{self.run_id}'

    def log_params(self, params):
        for key, value in params.items():
            self.client.log_param(self.run_id, key, str(value))

    def log_metric(self, name, value, step=None):
        self.client.log_metric(self.run_id, name, float(value), step=step)

    def log_figure(self, figure, artifact_path):
        artifact_path = Path(artifact_path)
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = Path(tmp_dir) / artifact_path.name
            figure.savefig(local_path, bbox_inches='tight', dpi=140)
            artifact_dir = None if artifact_path.parent == Path('.') else str(artifact_path.parent)
            self.client.log_artifact(self.run_id, str(local_path), artifact_path=artifact_dir)

    def end(self):
        _mlflow.end_run()


def describe_network(model):
    parts = []
    for name, module in model.named_modules():
        if name == '':
            continue
        if isinstance(module, torch.nn.Linear):
            parts.append(f'{name}: Linear({module.in_features}, {module.out_features})')
        elif len(list(module.children())) == 0:
            parts.append(f'{name}: {module.__class__.__name__}')
    return {
        'model_class': model.__class__.__name__,
        'architecture': ' | '.join(parts),
    }


def describe_loss(loss):
    return {
        'loss_function': loss.__class__.__name__,
    }


def describe_optimizer(optimizer):
    group = optimizer.param_groups[0]
    return {
        'optimizer': optimizer.__class__.__name__,
        'learning_rate': group.get('lr'),
        'weight_decay': group.get('weight_decay'),
        'betas': group.get('betas'),
    }


def describe_scheduler(scheduler):
    if scheduler is None:
        return {
            'scheduler': None,
        }
    return {
        'scheduler': scheduler.__class__.__name__,
        'scheduler_last_epoch': scheduler.last_epoch,
    }


def describe_dataloader(dataloader, name):
    return {
        f'{name}_batches': len(dataloader),
        f'{name}_samples': len(dataloader.dataset),
    }


def collect_training_params(
        run_name,
        model,
        loss,
        optimizer,
        scheduler,
        batch_size,
        n_epochs,
        train_dl,
        valid_dl):
    return {
        'run_name': run_name,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        **describe_network(model),
        **describe_loss(loss),
        **describe_optimizer(optimizer),
        **describe_scheduler(scheduler),
        **describe_dataloader(train_dl, 'train'),
        **describe_dataloader(valid_dl, 'valid'),
    }


def prepare_mlflow_logger(
        run_name,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        tracking_dir=MLFLOW_TRACKING_DIR,
        ui_base_url=MLFLOW_UI_BASE_URL):
    if not MLFLOW_AVAILABLE:
        _display_markdown('MLflow is not installed. Install it with `pip install mlflow` to enable experiment logging.')
        return NoOpMLflowLogger()

    tracking_dir = Path(tracking_dir).resolve()
    _mlflow.set_tracking_uri(tracking_dir.as_uri())
    _mlflow.set_experiment(experiment_name)
    ui_is_running = ensure_mlflow_ui(
        tracking_dir=tracking_dir,
        ui_base_url=ui_base_url,
    )
    if _mlflow.active_run() is not None:
        _mlflow.end_run()
    run = _mlflow.start_run(run_name=run_name)
    logger = MLflowRunLogger(run, ui_base_url=ui_base_url)

    if ui_is_running:
        _display_markdown(f'[Open MLflow run with graphs]({logger.url})')
    else:
        _display_markdown(
            f'MLflow could not start the UI automatically. Runs are stored locally in `{tracking_dir}`.'
        )
    return logger


def plot_training_curves(logger, history, title):
    if isinstance(history, dict):
        train_loss_history = history['train_loss']
        valid_loss_history = history['valid_loss']
        train_metrics_history = history.get('train_metrics', {})
        valid_metrics_history = history.get('valid_metrics', {})
    else:
        train_loss_history, train_acc_history, valid_loss_history, valid_acc_history = history
        train_loss_history = {'loss': train_loss_history}
        valid_loss_history = {'loss': valid_loss_history}
        train_metrics_history = {'accuracy': train_acc_history}
        valid_metrics_history = {'accuracy': valid_acc_history}

    if not isinstance(train_loss_history, dict):
        train_loss_history = {'loss': train_loss_history}
    if not isinstance(valid_loss_history, dict):
        valid_loss_history = {'loss': valid_loss_history}

    for loss_name, train_loss_values in train_loss_history.items():
        valid_loss_values = valid_loss_history.get(loss_name, [])
        if train_loss_values and valid_loss_values:
            epochs = range(1, len(train_loss_values) + 1)
            loss_fig, loss_ax = plt.subplots(figsize=(7, 4))
            loss_ax.plot(epochs, train_loss_values, marker='o', label='train')
            loss_ax.plot(epochs, valid_loss_values, marker='o', label='validation')
            loss_ax.set_title(f'{title}: {loss_name}')
            loss_ax.set_xlabel('epoch')
            loss_ax.set_ylabel(loss_name)
            loss_ax.grid(True, alpha=0.3)
            loss_ax.legend()
            logger.log_figure(loss_fig, f'figures/{loss_name}_curves.png')
            plt.show()
            plt.close(loss_fig)

    for metric_name, train_metric_history in train_metrics_history.items():
        valid_metric_history = valid_metrics_history.get(metric_name, [])
        if train_metric_history and valid_metric_history:
            epochs = range(1, len(train_metric_history) + 1)
            metric_fig, metric_ax = plt.subplots(figsize=(7, 4))
            metric_ax.plot(epochs, train_metric_history, marker='o', label='train')
            metric_ax.plot(epochs, valid_metric_history, marker='o', label='validation')
            metric_ax.set_title(f'{title}: {metric_name}')
            metric_ax.set_xlabel('epoch')
            metric_ax.set_ylabel(metric_name)
            metric_ax.grid(True, alpha=0.3)
            metric_ax.legend()
            logger.log_figure(metric_fig, f'figures/{metric_name}_curves.png')
            plt.show()
            plt.close(metric_fig)
