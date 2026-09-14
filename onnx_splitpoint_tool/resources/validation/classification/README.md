# Classification validation presets

This folder holds suite-portable classification validation subsets such as
`imagenet_val_mini_200` after they have been imported from a local ImageNet-1k
validation directory.

The tool does **not** ship ImageNet images. Use the importer utility to create a
local preset under `~/.onnx_splitpoint_tool/validation_datasets/classification/`
and the benchmark generator will copy the selected preset into a suite on demand.
