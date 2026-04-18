import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import MNIST
from model import ViT


def main():
	device = "mps" if torch.backends.mps.is_available() else "cpu"

	test_dataset = MNIST(is_train=False)
	test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

	model = ViT().to(device)
	model.load_state_dict(torch.load("model.pth", map_location=device))
	model.eval()

	total = 0
	correct = 0
	loss_sum = 0.0

	with torch.no_grad():
		for imgs, labels in test_loader:
			imgs = imgs.to(device)
			labels = labels.to(device)

			logits = model(imgs)
			loss = F.cross_entropy(logits, labels, reduction="sum")

			preds = logits.argmax(dim=1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)
			loss_sum += loss.item()

	avg_loss = loss_sum / total
	accuracy = correct / total

	print(f"Test samples: {total}")
	print(f"Test loss: {avg_loss:.6f}")
	print(f"Test accuracy: {accuracy:.4%}")


if __name__ == "__main__":
	main()
