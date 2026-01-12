from PIL import Image
import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 2

# Load the image
# img_path = "../../datasets/deepGlobe Land Cover Classification Dataset/train/split_image/10452_sat.jpg"
img_path = "../../datasets/test_dataset/kat_lal_bhak_tiles/output_785.png"
image = Image.open(img_path).convert("RGB")

# Load processor
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")

# Load base model and then load your fine-tuned checkpoint
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
).to(device)

# Load your fine-tuned weights
checkpoint_path = "segformer_only_road/checkpoints/best_model_fold1.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Preprocess image
inputs = feature_extractor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device)

# Forward pass
with torch.no_grad():
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits  # shape: [1, num_classes, H, W]

# Get predicted class per pixel
pred_mask = torch.argmax(logits.squeeze(), dim=0).cpu().numpy()

# Color map
colors = np.array([
    [0,255,255], #road
    [0,0,0],    #bg
], dtype=np.uint8)

seg_image = colors[pred_mask]

# Display
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(seg_image)
plt.title("Segmentation")
plt.axis("off")
plt.show()
