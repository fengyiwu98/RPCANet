from dulrs import dulrs_class
import torch

# Set CUDA as default device
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

dulrs = dulrs_class(
model_name="rpcanet_pp",
model_path="./result/ISTD/1K/RPCANet++_s6.pkl",     # Path for pretrained parameters
num_stages=6,
use_cuda=True)

# For heatmap generation
heatmap = dulrs.heatmap(
    img_path="./datasets/IRSTD-1k/test/images/000009.png",
    data_name="IRSTD-1k_test_images_000009",
    output_mat="./heatmap/mat",  # If users want to save the data as mat format. Default=None
    output_png="./heatmap/png"   # If users want to save the figure as png format. Default=None
)

# For lowrank calculation
lowrank_matrix = dulrs.lowrank_cal(
    img_path="./datasets/IRSTD-1k/test/images",
    model_name="rpcanet_pp",
    data_name="IRSTD-1k",
    save_dir= "./mats/lowrank"
)

# For lowrank paint based on calculation
lowrank_matrix_draw = dulrs.lowrank_draw(
    model_name="rpcanet_pp",
    data_name="IRSTD-1k",
    mat_dir= './mats/lowrank',
    save_dir = './mats/lowrank/figure' # Save path for result with png format
)

# For sparsity calculation
sparsity_matrix = dulrs.sparsity_cal(
    img_path="./datasets/IRSTD-1k/test/images",
    model_name="rpcanet_pp",
    data_name="IRSTD-1k",
    save_dir = './mats/sparsity'        # Save path for result with mat format
)