# medcore

Medical imaging utilities for DICOM / NIfTI workflows based on SimpleITK.

## Features
- IO
  - DICOM / NIfTI reading (`ImageReader`)
- Image utils
  - intensity array conversion / normalization
  - resampling / transform helpers
  - NIfTI write and label merge utilities
- Detection
  - coronal angle estimation
  - umbilicus detection helpers
- Segmentation
  - torso and abdomen segmentation classes
- Feature extraction
  - label area/volume summary
  - patch extraction around landmark points

## Installation
### From source
```bash
git clone https://github.com/yhchoi-data/medcore.git
cd medcore
pip install .
```

### From pypi [to be updated]
```bash
pip install medcore
```

## Quick start

```python
from medcore.io import ImageReader
from medcore.utils import (
    sitk_get_array,
    sitk_write_nii,
    sitk_read_labelfiles,
)

from medcore.detect import UmbilicusPredictor, UmbilicusDetector
from medcore.segment import TorsoSegmenter, AbdomenSegmenter
from medcore.feature import compute_label_volumes, extract_patches_from_image
```

## Package usage

```python
# IO
from medcore.io import ImageReader

# Utility functions
from medcore.utils import sitk_resampler, figure_overlay_label_on_slices

# Detection / segmentation
from medcore.detect import UmbilicusPredictor
from medcore.segment import TorsoSegmenter

# Feature extraction
from medcore.feature import compute_label_areas, compute_label_volumes
```

## Documentation
Detailed examples are in [USAGE.md](USAGE.md).
