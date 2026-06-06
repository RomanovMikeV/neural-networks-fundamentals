from pathlib import Path
import inspect
import subprocess


PYARMOR_PLATFORMS = (
    'windows.x86_64',
    'darwin.x86_64',
    'darwin.arm64',
    'linux.x86_64',
    'linux.aarch64',
)
DIST_KEEP_PATHS = (
    'task.py',
    'src',
    'pyarmor_runtime_000000',
    '.answer.json',
    '.solution.json',
    '.report.json',
)


def is_obfuscated(namespace=None):
    if namespace is None:
        frame = inspect.currentframe()
        namespace = frame.f_back.f_globals
    return '__pyarmor__' in namespace


def _remove_path(path):
    if path.is_dir():
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _clean_directory_by_whitelist(directory, keep_paths):
    directory = Path(directory)
    if not directory.exists():
        return

    keep_paths = {Path(path) for path in keep_paths}
    for path in directory.iterdir():
        relative_path = path.relative_to(directory)
        if relative_path not in keep_paths:
            _remove_path(path)


def build_task(
        notebook_path='task.ipynb',
        source_root='../../../../src',
        script_path='task.py',
        dist_dir='dist',
        dist_keep_paths=DIST_KEEP_PATHS,
        platforms=PYARMOR_PLATFORMS,
        namespace=None):
    if namespace is None:
        frame = inspect.currentframe()
        namespace = frame.f_back.f_globals

    if is_obfuscated(namespace):
        print('Running an obfuscated script.')
        caller_file = Path(namespace.get('__file__', '.')).resolve()
        _clean_directory_by_whitelist(caller_file.parent, dist_keep_paths)
        return

    print('Running outside of obfuscated script.')
    _clean_directory_by_whitelist(dist_dir, dist_keep_paths)

    platform_args = ','.join(platforms)
    subprocess.run(['jupyter', 'nbconvert', '--to', 'script', notebook_path], check=True)
    subprocess.run(['pyarmor', 'gen', '-r', source_root, '--platform', platform_args], check=True)
    subprocess.run(['pyarmor', 'gen', script_path, '--platform', platform_args], check=True)
    _remove_path(Path(script_path))
    _clean_directory_by_whitelist(dist_dir, dist_keep_paths)
    print(f'Obfuscated file created: {script_path}')
