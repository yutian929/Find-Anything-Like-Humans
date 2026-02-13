import torch
import alpha_clip
from PIL import Image
import numpy as np
from torchvision import transforms


class AlphaCLIP:
    def __init__(
        self,
        model_type="ViT-B/16",
        alpha_vision_ckpt_path="/home/yutian/projs/ASM/src/alpha_clip_pkg/scripts/weights/clip_b16_grit1m_fultune_8xe.pth",
        device=None,
    ):
        """
        Initialize AlphaCLIP

        Parameters:
            model_type: Model type, default is "ViT-B/16"
            alpha_vision_ckpt_path: Model checkpoint path
            device: Device to use, automatically selected if None
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_type = model_type

        # Determine image size based on model type
        self.mask_size = 336 if "@336px" in model_type else 224

        # Load model
        self.model, self.preprocess = alpha_clip.load(
            model_type, alpha_vision_ckpt_pth=alpha_vision_ckpt_path, device=self.device
        )

        # Prepare mask transformation
        self.mask_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.mask_size, self.mask_size)),
                transforms.Normalize(0.5, 0.26),
            ]
        )

    def process_mask(self, mask):
        """Process mask image and return transformed alpha mask

        Parameters:
            mask: Mask image path or loaded mask array
        """
        if isinstance(mask, str):
            mask = np.array(Image.open(mask))
        elif isinstance(mask, Image.Image):
            mask = np.array(mask)

        # Get binary mask array
        if len(mask.shape) == 2:
            binary_mask = mask == 255
        elif len(mask.shape) == 3:
            binary_mask = mask[:, :, 0] == 255
        else:
            raise ValueError("Unsupported mask format")

        alpha = self.mask_transform((binary_mask * 255).astype(np.uint8))
        alpha = alpha.half().to(self.device).unsqueeze(dim=0)
        return alpha

    def encode_images(self, rgb_image, masks):
        """Encode image and multiple mask regions

        Parameters:
            rgb_image: Image path or loaded PIL image
            masks: List of mask paths or loaded masks

        Returns:
            List of encoded image features, one feature vector per mask
        """
        # Process image input
        if isinstance(rgb_image, str):
            rgb_image = Image.open(rgb_image).convert("RGB")

        # Preprocess image
        processed_image = self.preprocess(rgb_image).unsqueeze(0).half().to(self.device)

        # Process mask input
        if not isinstance(masks, list):
            masks = [masks]

        # Process each mask and get features
        image_features_list = []
        with torch.no_grad():
            for mask in masks:
                alpha = self.process_mask(mask)
                image_features = self.model.visual(processed_image, alpha)
                # Normalize
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                image_features_list.append(image_features)

        return image_features_list

    def encode_texts(self, texts):
        """Encode text prompts

        Parameters:
            texts: List of text prompts

        Returns:
            Encoded text features
        """
        if isinstance(texts, str):
            texts = [texts]

        text = alpha_clip.tokenize(texts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text)

        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def ft_match_single_text_multi_images(self, text_features, image_features_list):
        """Calculate similarity between a single text and multiple image features

        Parameters:
            text_features: Text features
            image_features_list: List of image features

        Returns:
            List of similarities
        """
        similarities = []
        for image_features in image_features_list:
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            similarities.append(similarity.cpu().numpy())
        return similarities


# Usage example
if __name__ == "__main__":
    aclip = AlphaCLIP()

    img_path = "./examples/image.png"
    mask_paths = [
        "./examples/dress_mask.png",
        "./examples/flower_mask.png",
    ]  # Multiple masks example
    texts = [
        "a goegously dressed woman",
        "a purple sleeveness dress on a woman",
        "a purple sleeveness dress on a man",
        "bouquet of pink flowers",
        "Bouquet of flowers in woman's hand",
        "Bouquet of flowers in a man's hand",
    ]

    # Get image features
    image_features_list = aclip.encode_images(img_path, mask_paths)
    print(f"Encoded {len(image_features_list)} image regions")

    # Get text features
    text_features = aclip.encode_texts(texts)
    print(f"Encoded {len(texts)} text prompts")

    # Get similarities
    for i, image_features in enumerate(image_features_list):
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print(f"Label probs for mask {i+1}:", similarity.cpu().numpy())
