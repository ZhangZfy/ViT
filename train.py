import csv
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MNIST
from model import ViT

def main():
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu" 

    dataset=MNIST() # 数据集

    model = ViT().to(DEVICE) # 模型

    try:    # 加载模型
        model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    except:
        pass 

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # 优化器

    '''
        训练模型
    '''

    EPOCH = 50
    BATCH_SIZE = 64   # 从batch内选出10个不一样的数字

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, persistent_workers=True)    # 数据加载器

    os.makedirs("logs", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    csv_path = os.path.join("logs", "train_log.csv")
    csv_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["epoch", "iter", "loss", "lr"])

    iter_count = 0
    try:
        for epoch in range(EPOCH):
            for imgs, labels in dataloader:
                logits = model(imgs.to(DEVICE))

                loss = F.cross_entropy(logits, labels.to(DEVICE))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                current_lr = optimizer.param_groups[0]["lr"]

                writer.add_scalar("train/loss", loss_value, iter_count)
                writer.add_scalar("train/lr", current_lr, iter_count)
                csv_writer.writerow([epoch, iter_count, loss_value, current_lr])

                if iter_count % 1000 == 0:
                    print("epoch:{} iter:{} loss:{:.6f}".format(epoch, iter_count, loss_value))
                    csv_file.flush()
                    writer.flush()
                    torch.save(model.state_dict(), ".model.pth")
                    os.replace(".model.pth", "model.pth")
                iter_count += 1
    finally:
        csv_file.close()
        writer.close()

if __name__ == "__main__":
    main()