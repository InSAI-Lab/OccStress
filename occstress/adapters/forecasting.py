"""Build native model subprocess commands from the formal contract."""
import json
import os
from occstress.datasets.paths import repo_root, resolve_occstress_path
from occstress.distribution import require_source

ROOT = repo_root()

def command_plan(model, dataset, python, native_args, stage='world', root=ROOT, environ=None):
    env = dict(os.environ if environ is None else environ)
    contract = json.loads((root / 'configs/evaluation_contract.json').read_text())
    spec = contract['models'][model]
    require_source(root, spec['directory'])
    for name, forbidden in contract['legacy']['excluded_env_values'].items():
        if env.get(name, '').lower() in forbidden:
            raise ValueError(f'{name}={env[name]} selects legacy temporal alignment; unset it')
    blocked = ('--config', '--py-config', '--world-config', '--tokenizer-config', '--cfg-options', '--dataset')
    if any(argument.split('=', 1)[0] in blocked for argument in native_args):
        raise ValueError('Formal config/dataset overrides are not allowed; use the native entrypoint for exploratory runs')
    directory = root / spec['directory']
    config = str(directory / spec['configs'][dataset])
    entrypoint = str(directory / spec['entrypoints'][dataset])
    if model == 'iiworld' and dataset == 'nuscenes':
        if stage == 'tokenizer':
            config = str(directory / spec['tokenizer_configs'][dataset])
        command = [python, entrypoint, config, *native_args]
    elif model == 'iiworld':
        if stage != 'world':
            raise ValueError('Cross-dataset wrapper runs tokenizer and world model together')
        command = [python, entrypoint, '--world-config', config, '--tokenizer-config',
                   str(directory / spec['tokenizer_configs'][dataset]), '--dataset', dataset, *native_args]
    else:
        if stage != 'world':
            raise ValueError('--stage tokenizer only applies to II-World nuScenes')
        config_flag = '--py-config' if dataset == 'nuscenes' and model in {'occworld', 'come'} else '--config'
        command = [python, entrypoint, config_flag, config, *native_args]
    # Do not inherit another model's PYTHONPATH or stale cross-dataset selectors.
    env['PYTHONPATH'] = os.pathsep.join([str(directory), str(root),
                                       str(root / 'scripts/waymo'), str(root / 'scripts/carla')])
    env['PYTHONNOUSERSITE'] = '1'
    env['OCCSTRESS_CODE_ROOT'] = str(root)
    env['OCCSTRESS_DATASET'] = dataset
    env['OMP_NUM_THREADS'] = env.get('OMP_NUM_THREADS', '1')
    for name in list(env):
        if (name.startswith('OCCSTRESS_WAYMO_') and dataset != 'waymo') or (
                name.startswith('OCCSTRESS_CARLA_') and dataset != 'carla'):
            env.pop(name)
    variable = contract['datasets'][dataset]['root_env']
    # data_root reads the real process env; honor the supplied env explicitly.
    env[variable] = env.get(variable, str(root / 'data' / 'OccStress'))
    # Native wrappers change their cwd. Resolve only dataset arguments here;
    # checkpoints and output destinations retain their native semantics.
    for index, argument in enumerate(command):
        flag, separator, value = argument.partition('=')
        if flag not in {'--protocol', '--protocol-path', '--base-info'}:
            continue
        if not separator:
            if index + 1 >= len(command):
                raise ValueError(f'Missing value for {flag}')
            value = command[index + 1]
        resolved = str(resolve_occstress_path(value, code_root=root,
                       occstress_root=env[variable], dataset=dataset,
                       external_root=env.get('OCCSTRESS_EXTERNAL_ROOT')))
        if separator:
            command[index] = flag + '=' + resolved
        else:
            command[index + 1] = resolved
    env.update(spec['required_env'])
    return directory, command, env
