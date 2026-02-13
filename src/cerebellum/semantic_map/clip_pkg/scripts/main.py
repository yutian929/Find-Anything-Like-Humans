import torch
import clip
from PIL import Image
import numpy as np
import time  # Added for timing

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)


def calculate_similarity(text_feature, image_feature):
    """计算文本特征与单张图片特征的相似度"""
    text_feature = np.array(text_feature)
    image_feature = np.array(image_feature)
    similarity = np.dot(image_feature, text_feature) / (
        np.linalg.norm(image_feature) * np.linalg.norm(text_feature)
    )
    return similarity


image_paths = ["cat.jpg", "dog.jpg"]
text = "dog"
images = []
for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    images.append(image)
batch = torch.stack(images).to("cuda")
text_tokens = clip.tokenize([text]).to(device)

with torch.no_grad():
    iterations = 5  # Number of iterations to average timing
    image_encode_times = []
    text_encode_times = []

    for _ in range(iterations):
        # Measure encode_image time
        start_time = time.time()
        image_features = model.encode_image(batch).cpu().numpy()
        image_encode_times.append(time.time() - start_time)

        # Measure encode_text time
        start_time = time.time()
        text_features = model.encode_text(text_tokens).cpu().numpy()
        text_encode_times.append(time.time() - start_time)

    avg_image_encode_time = sum(image_encode_times) / iterations
    avg_text_encode_time = sum(text_encode_times) / iterations

    print(
        f"Average encode_image time over {iterations} iterations: {avg_image_encode_time:.4f} seconds"
    )
    print(
        f"Average encode_text time over {iterations} iterations: {avg_text_encode_time:.4f} seconds"
    )

    # 计算相似度
    for i in range(len(image_paths)):
        similarity = calculate_similarity(text_features[0], image_features[i])
        print(
            f"Similarity between text {text} and image {image_paths[i]}: {similarity:.4f}"
        )

# Average encode_image time over 5 iterations: 0.0189 seconds
# Average encode_text time over 5 iterations: 0.0065 seconds
# Similarity between text dog and image cat.jpg: 0.2286
# Similarity between text dog and image dog.jpg: 0.2908
