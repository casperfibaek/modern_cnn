# modern_cnn
source .venv/bin/activate

# Set torch settings:
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # also silences the torch.compile warning

# Structure
/datasets/imagenette2/....
/checkpoints/....

# GRID Drivers necessary for shared vGPUs