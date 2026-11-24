import json
import os
from pathlib import Path

root = Path(os.environ.get('LIBERO_DATA_ROOT', 'data/libero')) / 'libero_spatial_no_noops_lerobot'
info = json.loads((root / 'meta/info.json').read_text())
features = info['features']
required = {'observation.images.image', 'observation.images.wrist_image', 'observation.states.joint_state', 'observation.states.ee_state', 'observation.states.gripper_state', 'action'}
missing = required - features.keys()
for key in sorted(required):
    print(key, features.get(key, {}).get('shape', 'MISSING'))
if missing:
    raise SystemExit(f'Incompatible dataset: missing {sorted(missing)}')
for key, size in [('action', 7), ('observation.states.joint_state', 7), ('observation.states.ee_state', 6), ('observation.states.gripper_state', 2)]:
    if list(features[key].get('shape', [])) != [size]:
        raise SystemExit(f'Unexpected shape: {key}')
print('[SPATIAL] Metadata check passed:', root)
