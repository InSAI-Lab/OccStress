#!/usr/bin/env python3
"""Print (or explicitly execute) an isolated model environment recipe."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from occstress.distribution import require_source


def install_plan(model, prefix, cuda_home, arch, conda='conda', root=ROOT,
                 conda_channel='conda-forge', cc=None, cxx=None, clean_build_flags=False):
    registry = json.loads((root / 'environments/profiles.json').read_text())
    spec = registry['models'][model]
    require_source(root, spec['directory'])
    recipe = registry['recipes'][spec['recipe']]
    if spec.get('manual_install_only'):
        raise ValueError(f'{model}: see docs/ENVIRONMENTS.md for the legacy native installation')
    python = str(prefix / 'bin/python')
    pip = [python, '-m', 'pip']
    env = {
        'PYTHONNOUSERSITE': '1', 'CUDA_HOME': str(cuda_home),
        'TORCH_CUDA_ARCH_LIST': arch, 'FORCE_CUDA': '1',
        'MMCV_WITH_OPS': '1', 'MAX_JOBS': '4',
        'OMP_NUM_THREADS': '1', 'MPLCONFIGDIR': str(prefix / 'cache/matplotlib'),
        'TORCH_EXTENSIONS_DIR': str(prefix / 'cache/torch-extensions'),
    }
    if cc:
        env['CC'] = cc
    if cxx:
        env['CXX'] = cxx
    if clean_build_flags:
        env.update({name: '' for name in ('CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS')})
    commands = []

    def add(command, cwd=root):
        commands.append({'cwd': str(cwd), 'command': command})

    add([conda, 'create', '-y', '--override-channels', '-c', conda_channel, '-p', str(prefix),
         'python=' + recipe['python'], 'pip'])
    add(pip + ['install', 'setuptools==69.5.1', 'wheel==0.45.1', 'ninja==1.13.0',
               'packaging==24.2', 'psutil==6.1.1'])
    add(pip + ['install', '--extra-index-url',
               'https://download.pytorch.org/whl/' + recipe['index'],
               'torch==' + recipe['torch'], 'torchvision==' + recipe['torchvision']])
    req = str(root / recipe['requirements'])
    # Install dependency pins before building MMCV. mmdet/mmseg do not fetch mmcv.
    add(pip + ['install', '-r', req, '-c', req])
    add(pip + ['install', '--no-build-isolation', '--no-deps',
               '--no-binary', recipe['mmcv_package'],
               recipe['mmcv_package'] + '==' + recipe['mmcv']])
    if spec.get('extra_requirements'):
        add(pip + ['install', '--no-build-isolation', *spec['extra_requirements'], '-c', req])
    directory = root / spec['directory']
    if spec.get('dependency_directory') and not spec.get('dependency_path_only'):
        add(pip + ['install', '--no-build-isolation', '--no-deps', '-e',
                   str(root / spec['dependency_directory'])])
    if spec['editable']:
        add(pip + ['install', '--no-build-isolation', '--no-deps', '-e', str(directory)])
    for build in spec['builds']:
        add([python, 'setup.py', 'build_ext', '--inplace'], directory / build)
    add(pip + ['check'])
    add([python, str(root / 'tools/check_environment.py'), '--model', model])
    return {'model': model, 'recipe': recipe, 'env': env, 'commands': commands,
            'gpu_reproduction_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--cuda-home', type=Path, required=True)
    parser.add_argument('--arch', required=True, help='E.g. 8.0 (A100), 9.0 (H100), 12.0 (5090).')
    parser.add_argument('--conda', default='conda')
    parser.add_argument('--conda-channel', default='conda-forge')
    parser.add_argument('--cc', help='CUDA-compatible host C compiler; overrides inherited CC.')
    parser.add_argument('--cxx', help='Matching host C++ compiler; overrides inherited CXX.')
    parser.add_argument('--clean-build-flags', action='store_true',
                        help='Clear inherited module-specific compiler/linker flags for this install only.')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    prefix = args.prefix.expanduser().resolve()
    try:
        plan = install_plan(args.model, prefix, args.cuda_home.resolve(), args.arch, args.conda,
                            conda_channel=args.conda_channel, cc=args.cc, cxx=args.cxx,
                            clean_build_flags=args.clean_build_flags)
    except ValueError as error:
        parser.error(str(error))
    for key, value in plan['env'].items():
        print(f'export {key}={shlex.quote(value)}')
    for step in plan['commands']:
        print(f'(cd {shlex.quote(step["cwd"])} && {shlex.join(step["command"])})')
    if not args.execute:
        print('# Dry run only. No environment or package has been changed.')
        return
    if prefix.exists():
        raise SystemExit('Refusing to modify an existing prefix; use a new model-specific prefix.')
    if not (args.cuda_home / 'bin/nvcc').is_file():
        raise SystemExit('CUDA toolkit with nvcc is required; PyTorch wheels alone do not supply it.')
    version_output = subprocess.check_output([str(args.cuda_home / 'bin/nvcc'), '--version'], text=True)
    version = re.search(r'release\s+(\d+\.\d+)', version_output)
    if not version or version.group(1) != plan['recipe']['cuda']:
        raise SystemExit('nvcc must match the recipe CUDA version: ' + plan['recipe']['cuda'])
    if float(args.arch.split(';')[0].replace('+PTX', '')) >= 10 and plan['recipe']['cuda'] != '12.8':
        raise SystemExit('The selected legacy recipe is not a Blackwell profile.')
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(plan['env'])
    env['PATH'] = str(args.cuda_home / 'bin') + os.pathsep + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = str(args.cuda_home / 'lib64') + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    for step in plan['commands']:
        subprocess.run(step['command'], cwd=step['cwd'], env=env, check=True)


if __name__ == '__main__':
    main()
