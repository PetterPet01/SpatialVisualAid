# Class related params
class_set = "ram"
add_bg_classes = False
accumu_classes = False
exp_suffix = None
rm_bg_classes = True
add_classes = []
remove_classes = [
    "room", "kitchen", "office", "house", "home", "building", "corner", "shadow", "carpet", "photo", "sea", "shade", "stall", "space", "aquarium", "apartment", "image", "city", "blue", "skylight", "hallway", "bureau", "modern", "salon", "doorway", "wall lamp", "scene", "sun", "sky", "smile", "cloudy", "comfort", "white", "black", "red", "green", "yellow", "purple", "pink", "stand", "wear", "area", "shine", "lay", "walk", "lead", "bite", "sing",
]
bg_classes = ["wall", "floor", "ceiling"]
sam_variant = "sam-hq"
specified_tags = "None"
box_threshold = 0.25
text_threshold = 0.2
nms_threshold = 0.5
masking_option = "none"
mask_area_threshold = 100
mask_conf_threshold = 0.3
max_bbox_area_ratio = 0.90
skip_bg = False

# Depth model configuration

unik3d_model_size = "Large"# Only use unik3d for depth estimation
yoloe_model_path="yoloe-11m-seg-pf.pt"
yoloe_confidence_threshold=0.45
min_mask_area_pixel=150
yoloe_mask_dilate_iterations=0
dbscan_remove_noise = False

sort_detections_by_area = False
# Point cloud processing params
min_points_threshold = 50
min_points_threshold_after_denoise = 25
bbox_min_volume_threshold = 1e-6 # Minimum volume for a bounding box to be considered valid
pcd_perturb_std = 0.001 # Standard deviation for random perturbation of points

pcd_sor_neighbors = 20 # Statistical Outlier Removal: number of neighbors
pcd_sor_std_ratio = 1.5 # Statistical Outlier Removal: std ratio
pcd_voxel_size = 0.015   # Voxel size for downsampling (e.g., 1cm). Set to 0 to disable.
obb_robust = True # Use robust OBB calculation in Open3D

downsample_voxel_size = 0.02
dbscan_remove_noise = True
dbscan_eps = 0.15
dbscan_min_points = 15
spatial_sim_type = "overlap"
log_dir = "./demo_output_generalized_hf/logs"
vis = True
use_clip = False
perspective_model_variant = "perspective_fields"
export_yoloe_debug_viz = True
export_ply_with_bboxes = True
image_resize_height = 384
debug_folder = "./demo_output_generalized_hf/debug"
debug_logs = "./demo_output_generalized_hf/captured"

# LLM related (Hugging Face)
# llm_model_name_hf = "Qwen/Qwen2.5-1.5B" # Smaller model for quick testing, e.g., Qwen/Qwen2-0.5B-Instruct or Qwen/Qwen2-1.5B-Instruct
llm_model_name_hf = "Qwen/Qwen2.5-1.5B-Instruct" # Larger model, requires more resources & HF_TOKEN
llm_max_retries = 2
llm_temperature = 0.1
llm_max_new_tokens = 1024 * 10 # Reduced for smaller models, adjust as needed