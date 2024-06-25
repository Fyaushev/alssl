from pathlib import Path

from dpipe.io import load, save
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

DATA_PATH = Path("/shared/projects/active_learning/data/cifar10")

if __name__ == "__main__":
    save_root = Path("/shared/projects/active_learning/embeddings/cifar10/dinov2")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")

    for path in tqdm(DATA_PATH.glob("*/*/*.png")):
        save_path = save_root / path.parts[-3] / path.parts[-2]
        save_path.mkdir(parents=True, exist_ok=True)

        image = load(path)
        inputs = processor(images=image, return_tensors="pt")
        output = model(**inputs)
        embedding = output.pooler_output.detach().numpy()

        save(embedding, save_path / (path.parts[-1].split(".")[0] + ".npy"))
