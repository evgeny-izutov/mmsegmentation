# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project is based on [mmsegmentation project by OpenMMLab](https://github.com/open-mmlab/mmsegmentation).
With respect to it we made the following changes.

## \[2022-05-09\]
### Added
* HPO support
* Feature dump support
* NNCF support
* Lite-HRNet-18-mod2 (middle)
* Lite-HRNet-x-mod3 (heavy)

### Removed
* Tasks & model templates (moved to OTE)


## \[2021-12-27\]
### Added
- Support of datasets: `COCO Stuff`, `Kvasir-Seg`, `Kvasir-Instrument`.
- Implemented `MaskCompose` and `ProbCompose` composers to merge different
  augmentation pipelines.
- Implemented augmentations: `MixUp`, `CrossNorm`.
- Support of the pixel-weighting method to focus training on class borders.
- Support of backbone architectures (including the appropriate config files):
  `BiSeNet V2`, `CABiNet`, `DABNet`, `DDRNet`, `EfficientNet`, `ICNet`, `ShelfNet`,
  `STDCNet`, `Lite-HRNet`.
- Support of head architectures: `BiSeHead`, `DDRHead`, `HamburgerHead`,
  `HyperSegHead`, `ICHead`, `MemoryHead`, `ShelfHead`.
- Support of `EMA` hook.
- MMSegmentation can now use custom optimizer hook with `Adaptive Gradient Clipping`
  and custom learning rate hooks (`cos`, `step`) with support of three-stage
  training: `freeze`, `warm-up` and `default`.
- Support of loss miners: `ClassWeightingPixelSampler`, `LossMaxPooling`.
- Implemented `AngularPWConv` layer to support ML-based heads.
- Implemented `LocalContrastNormalization` layer to normalize the input of a network.
- Implemented loss factory which supports the following pixel-level losses:
  `CrossEntropy`, `CrossEntropySmooth`, `NormalizedCrossEntropy`,
  `ReverseCrossEntropy`, `SymmetricCrossEntropy`, `ActivePassiveLoss`.
- Implemented `Tversky` and `Boundary` losses.
- Implemented module to freeze the pattern-matched layers during training.
- Export to InferenceEngine format which allow to run on edge-oriented devices.
- Integration of NNCF model optimization.
- Support of OTE tasks which allow to run the following commands through the API:
  `train`, `eval`, `export`, `optimization`.
- Implemented scalar schedulers for ML-related scalar values (e.g. scale,
  regularization weight, loss weight): `constant`, `step`, `poly`.
- Implemented script `init_venv.sh` to initialize the whole mmsegmentation-related
  environment.
- Support of CPU-only training mode.


### Changed
- The following datasets have been updated to support MaskCompose augmentation:
  `ADE20k`, `CHASE`, `Cityscapes`, `Drive`, `HRF`, `Stare`, `Pascal VOC12`,
  `Pascal VOC12 Aug`.
- Unified the head architectures: `FCNHead`, `DepthwiseSeparableFCNHead`.
- Updated `OCRHead` to support depthwise separable convolutions.
- Updated `OHEM` loss miner to support `valid ratio` hyperparameter.
- Refactored the base network class to support a set of losses per head with
  adaptive loss re-weighting.
- Refactored base loss class to support: `loss-independent pixel miners`, `PR-product`,
  `MaxEntropy` regularization, `pixel-level losses re-weighting` according to the
  weight mask, `loss-jitter` regularization.
- Unified `CrossEntropy` and `Dice` losses.
- Updated `Dice` loss to support `General Dice` and `Dice++` losses.
- Updated `tools/export.py` for the model export to support the implemented
  API-based export method.
- Added support of `fvcore` in `tools/get_flops.py` tool.

### Deprecated
- TBD

### Removed
- TBD

### Fixed
- TBD

### Security
- TBD
