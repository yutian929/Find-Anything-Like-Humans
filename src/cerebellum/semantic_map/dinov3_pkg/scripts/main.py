import sys
import torch
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from PIL import Image
from dinov3.data.transforms import make_classification_eval_transform
import torch.nn.functional as F
import time  # Added for timing


# Load pretrained DINOv3 vision head and text model
model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
model = model.cuda()
# model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().cuda()
# tokenizer = get_tokenizer()

# Load multiple images from local files
def load_images_from_files(file_paths: list) -> list:
    images = []
    for file_path in file_paths:
        images.append(Image.open(file_path).convert("RGB"))
    return images


# Local image file paths
LOCAL_IMAGE_PATHS = ["cat.jpg", "dog.jpg"]
images_pil = load_images_from_files(LOCAL_IMAGE_PATHS)

# Preprocess images and tokenize text
image_preprocess = make_classification_eval_transform()
image_tensors = torch.stack([image_preprocess(img) for img in images_pil], dim=0).cuda()
text = "dog"
tokenized_text_tensor = tokenizer.tokenize([text]).cuda()

# Compute similarity between image features and text feature
with torch.autocast("cuda", dtype=torch.float):
    with torch.no_grad():
        iterations = 5  # Number of iterations to average timing
        image_encode_times = []
        text_encode_times = []

        for _ in range(iterations):
            # Measure encode_image time
            start_time = time.time()
            image_features = model.encode_image(image_tensors)  # Shape: [N, D]
            image_encode_times.append(time.time() - start_time)
            # breakpoint()
            # Measure encode_text time
            start_time = time.time()
            text_features = model.encode_text(tokenized_text_tensor)  # Shape: [1, D]
            text_encode_times.append(time.time() - start_time)

        avg_image_encode_time = sum(image_encode_times) / iterations
        avg_text_encode_time = sum(text_encode_times) / iterations

        print(
            f"Average encode_image time over {iterations} iterations: {avg_image_encode_time:.4f} seconds"
        )
        print(
            f"Average encode_text time over {iterations} iterations: {avg_text_encode_time:.4f} seconds"
        )

image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize image features
text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize text feature

# Compute similarity: [N, D] @ [D, 1] -> [N, 1]
similarity = (image_features @ text_features.T).squeeze(-1).cpu().numpy()

# Print similarity scores for each image
for i, score in enumerate(similarity):
    print(f"Image {LOCAL_IMAGE_PATHS[i]} similarity with text '{text}': {score}")

# Average encode_image time over 5 iterations: 0.0178 seconds
# Average encode_text time over 5 iterations: 0.0520 seconds
# Image cat.jpg similarity with text 'dog': 0.09666256606578827
# Image dog.jpg similarity with text 'dog': 0.1321798413991928
