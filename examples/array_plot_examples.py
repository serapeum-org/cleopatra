import numpy as np
import numpy.ma as ma

from cleopatra.config import Config

Config.set_matplotlib_backend()
import matplotlib.pyplot as plt

from cleopatra.array_glyph import ArrayGlyph

# from matplotlib.transforms import blended_transform_factory
# %% create the glyph from a masked array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
arr = ma.masked_array(arr, mask=mask)
array = ArrayGlyph(arr)
# array.plot()
# %%
arr = np.load("tests/data/s2a.npy")
arr = np.moveaxis(arr, 0, -1)
arr2 = arr[:, :, [3, 2, 1]]
rgb = np.clip(arr2 / 10000, 0, 1)
plt.imshow(rgb)
plt.show()
# %%
# plt.ioff()
array = ArrayGlyph(arr, rgb=[3, 2, 1], cutoff=[0.3, 0.3, 0.3])
# %%
arr = np.load("tests/data/arr.npy")
exclude_value = arr[0, 0]
cmap = "terrain"
# arr2 = np.load("tests/data/DEM5km_Rhine_burned_fill.npy")
# exclude_value2 = arr2[0, 0]
color_scale = ["linear", "power", "sym-lognorm", "boundary-norm", "midpoint"]
ticks_spacing = 10


midpoint = 10
array = ArrayGlyph(arr, exclude_value=[exclude_value])
array.plot(
    color_scale=color_scale[4],
    midpoint=midpoint,
    cmap=cmap,
    ticks_spacing=ticks_spacing,
)
# %%
array = ArrayGlyph(arr, exclude_value=[exclude_value])
fig, ax = array.plot(
    cbar_orientation="vertical",
    cbar_label_rotation=-90,
    cbar_label_location="center",
    cbar_length=0.8,
    cbar_label_size=10,
    cbar_label="Discharge m3/s",
    color_scale="linear",
    cmap="coolwarm_r",
)
# %% test_plot_array_color_scale_1
array = ArrayGlyph(arr, exclude_value=[exclude_value])
array.plot(
    color_scale=color_scale[0],
    cmap=cmap,
    ticks_spacing=ticks_spacing,
)
# %% test_plot_array_color_scale_2
color_scale_2_gamma = 0.5
array = ArrayGlyph(arr, exclude_value=[exclude_value])
array.plot(
    color_scale=color_scale[1],
    cmap=cmap,
    gamma=color_scale_2_gamma,
    ticks_spacing=ticks_spacing,
)
# %% test_plot_array_color_scale_3
ticks_spacing = 5
color_scale_3_linscale = 0.1
color_scale_3_linthresh = 0.0001
color_scale_3_linthresh = 0.015
array = ArrayGlyph(arr, exclude_value=[exclude_value])
array.plot(
    color_scale=color_scale[2],
    line_scale=color_scale_3_linscale,
    line_threshold=color_scale_3_linthresh,
    cmap=cmap,
    ticks_spacing=ticks_spacing,
)
# %% test_plot_array_color_scale_4
ticks_spacing = 10
array = ArrayGlyph(arr, exclude_value=[exclude_value])
array.plot(
    color_scale=color_scale[3],
    cmap=cmap,
    ticks_spacing=ticks_spacing,
)
# %%
# bounds = [-559, 0, 440, 940, 1440, 1940, 2440, 2940, 3500]
# bounds = [0,  440,  940, 1440, 1940, 2440, 2940, 3500]
bounds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
array = ArrayGlyph(arr, exclude_value=[exclude_value])
array.plot(
    color_scale=color_scale[3],
    bounds=bounds,
    cmap=cmap,
    ticks_spacing=ticks_spacing,
)

# %% test_plot_array_color_scale_5
midpoint = 10
array = ArrayGlyph(arr, exclude_value=[exclude_value])
array.plot(
    color_scale=color_scale[4],
    midpoint=midpoint,
    cmap=cmap,
    ticks_spacing=ticks_spacing,
)
# %% test_plot_array_display_cell_values
display_cell_value = True
num_size = 8
background_color_threshold = None
array = ArrayGlyph(arr, exclude_value=[exclude_value])
array.plot(
    display_cell_value=display_cell_value,
    num_size=num_size,
    background_color_threshold=background_color_threshold,
    ticks_spacing=ticks_spacing,
)
# %%
coello_data = np.load("tests/data/coello.npy")
exclude_value = arr[0, 0]
animate_time_list = list(range(1, 11))
array = ArrayGlyph(coello_data, exclude_value=[exclude_value])
anim = array.animate(
    animate_time_list, title="Flow Accumulation", display_cell_value=True
)

import sys
from pathlib import Path

# path = Path(sys.executable)
# f"{path.parent}/{}"
# %%

# amin.save('examples/animation.gif', fps=2)
video_format = "mp4"

import matplotlib as mpl

fps = 2

path = "examples/animation.mp4"
import ffmpeg
