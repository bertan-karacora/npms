from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Setup
filename_data = "partial_sdf_grd_256.npz"

dir_datasets = "datasets"
dir_set = "cape_single"
dir_identity = "00032_shortshort"
dir_sequence = "chicken_wings"
dir_frame = "000022"
dir_processed_data = "partial_sdf_grd"

path_set = Path(".") / dir_datasets / dir_set
path_frame = path_set / dir_identity / dir_sequence / dir_frame
path_data = path_frame / dir_processed_data / filename_data

# Run
data = np.load(path_data)
sdf = data["sdf_grid"].reshape(256, 256, 256)
mask = data["validity_mask"].reshape(256, 256, 256)
sdf_masked = mask.astype(np.float32) * sdf
z = 130
slice = np.rot90(sdf_masked[..., z])

fig = plt.figure()
plt.imshow(slice)
plt.axis("off")
plt.savefig(f"sdf_slice_{z}.png", bbox_inches="tight")
