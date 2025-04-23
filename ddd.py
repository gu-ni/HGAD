# %%
import os
import pandas as pd
import json

path = '/workspace/meta_files/base_classes.json'
df = pd.read_json(path)
# %%
df
# %%
class_to_idx = {}
for i, x in enumerate(df['train'].keys()):
    class_to_idx[x] = i

# %%
class_to_idx
# %%
save = '/workspace/MegaInspection/HGAD/outputs/scenario_1/base/class_mapping_base.json'
with open(save, 'w') as f:
    json.dump(class_to_idx, f, indent=4)
# %%
329/542
# %%
