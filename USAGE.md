# Usage

This document describes common usage patterns for **medcore**.

## 1. Imports

```python
from medcore.io import ImageReader

from medcore.utils import (
    sitk_get_array,
    sitk_write_nii,
    sitk_make_euler3dtransform,
    sitk_resampler,
    sitk_read_labelfiles,
)

from medcore.detect import UmbilicusPredictor, UmbilicusDetector
from medcore.segment import TorsoSegmenter, AbdomenSegmenter
from medcore.feature import (
    compute_label_volumes,
    compute_label_areas,
    extract_patches_from_image,
)
```

## 2. Load image (DICOM / NIfTI)

```python
from medcore.io import ImageReader

vol = ImageReader("/path/to/image.nii.gz").read()
print(vol.GetSize(), vol.GetSpacing())
```

```python
vol = ImageReader("/path/to/dicom_dir").read()
```

## 3. Convert to NumPy and normalize

```python
from medcore.utils import sitk_get_array

arr = sitk_get_array(vol)  # raw array
arr_norm = sitk_get_array(vol, normalize=True, norm_min=-500, norm_max=2000)
```

## 4. Resampling and transforms

```python
from medcore.utils import sitk_make_euler3dtransform, sitk_resampler

tfm = sitk_make_euler3dtransform(vol, rotation_deg=15, axis="x")
vol_rot = sitk_resampler(vol, transform=tfm, interpolation="linear")
vol_iso = sitk_resampler(vol, new_spacing=(1.0, 1.0, 1.0))
```

## 5. Save NIfTI

```python
from medcore.utils import sitk_write_nii, sitk_get_array

img = sitk_get_array(vol)
# ... processing ...
sitk_write_nii(vol, "/path/to/out_volume.nii.gz")
sitk_write_nii(img, "/path/to/out_array.nii.gz", reference=vol)
```

## 6. Segmentation

```python
from medcore.segment import TorsoSegmenter, AbdomenSegmenter

torso_seg = TorsoSegmenter()
torso_mask, torso_contour, torso_smooth = torso_seg.segment(arr_norm)

abd_seg = AbdomenSegmenter()
abdominal_image, abdomen_mask, abdomen_contour = abd_seg.segment(arr_norm)
```

## 7. Detection

```python
from medcore.detect import UmbilicusPredictor

predictor = UmbilicusPredictor()
point_xyz = predictor.predict(arr, spacing=vol.GetSpacing())
print(point_xyz)
```

```python
from medcore.detect import UmbilicusDetector

detector = UmbilicusDetector()
points_df = detector.detect(
    region_image=abdominal_image,
    region_mask=abdomen_mask,
    region_contour=abd_seg.contour_info,
    region_info=abd_seg.abdomen_region,
)
print(points_df.head())
```

## 8. Feature extraction

```python
from medcore.feature import compute_label_volumes, compute_label_areas

labelfiles = {
    1: "/path/to/muscle.nii.gz",
    2: "/path/to/fat.nii.gz",
}

volumes_cm3 = compute_label_volumes(labelfiles)
areas_cm2 = compute_label_areas(labelfiles, slices_index=100)
print(volumes_cm3)
print(areas_cm2)
```

```python
from medcore.feature import extract_patches_from_image

# points: (N, 3), commonly 25 points for 5x5 grid
patches = extract_patches_from_image(points, vol, patch_size=50, middle_size=50, delta=25)
print(patches.shape)
```

## 9. Label merge utility

```python
from medcore.utils import sitk_read_labelfiles

merged = sitk_read_labelfiles(labelfiles)  # labels merged into one UInt8 volume
```

## 10. Notes

- `medcore.segment` import is supported directly:
  - `from medcore.segment import TorsoSegmenter`
- `compute_label_volumns` is still available as a backward-compatible alias of `compute_label_volumes`.
