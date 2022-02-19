from dataset import SHREC2022Primitives, minkowski_collate
import transforms as t
import torch
from torch.utils.data import DataLoader
from networks import MinkowskiFCNN
import MinkowskiEngine as ME
from tqdm import tqdm
import os
path = "/home/ioannis/Desktop/programming/data/SHREC/SHREC2022/dataset"

train_transforms = [t.Translate(), 
                    t.SphereNormalization(), 
                    t.Initialization(),
                    t.RandomRotate(180, 0),
                    t.RandomRotate(180, 1),
                    t.RandomRotate(180, 2),
                    t.GaussianNoise()]

valid_transforms = [t.Translate(), 
                    t.SphereNormalization()]

t_dataset = SHREC2022Primitives(path, train=True, valid=False, valid_split=0.2, transform=train_transforms)
v_dataset = SHREC2022Primitives(path, train=True, valid=True, valid_split=0.2, transform=valid_transforms)

batch_size = 256
train_loader = DataLoader(t_dataset, batch_size=batch_size, shuffle=True, collate_fn=minkowski_collate, num_workers=8)
eval_loader = DataLoader(v_dataset, batch_size=batch_size, shuffle=False, collate_fn=minkowski_collate, num_workers=8)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

net = MinkowskiFCNN(in_channel = 3, out_channel = 5).to(device)

checkpoint_path = "/home/ioannis/Desktop/programming/phd/SHREC/SHREC2022/checkpoints"
checkpoint_file = os.path.join(checkpoint_path, "checkpoint.pth")
net.load_state_dict(torch.load(checkpoint_file))

def create_input_batch(batch, device="cuda", quantization_size=0.05):
    batch["coordinates"][:, 1:] = batch["coordinates"][:, 1:] / quantization_size
    return ME.TensorField(
        coordinates=batch["coordinates"],
        features=batch["features"],
        device=device
    )


optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# 
num_epochs = 100
eval_step = 2
for epoch in range(num_epochs):
    # for loss and accuracy tracking the training set
    m_loss = 0
    acc = 0
    for batch in tqdm(train_loader):

        optimizer.zero_grad()

        labels = batch["labels"].long().to(device)
        batch_size = labels.shape[0]

        minknet_input = create_input_batch(
            batch, 
            device=device,
            quantization_size=0.05
        )

        pred = net(minknet_input)

        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        m_loss += loss.item()
        acc += (torch.max(pred, dim=-1).indices == labels).sum().item() / batch_size

    m_loss /= len(train_loader)
    acc /= len(train_loader)

    print(f" Epoch: {epoch} | Training: loss = {m_loss} || accuracy = {acc*100}%")

    if (epoch+1) % eval_step == 0:
        acc = 0
        m_loss = 0
        net.eval()
        with torch.no_grad():
            for batch in tqdm(eval_loader):

                labels = batch["labels"].long().to(device)
                batch_size = labels.shape[0]

                net_in = create_input_batch(
                    batch, 
                    device=device, 
                    quantization_size=0.05
                )

                pred = net(net_in)

                loss = criterion(pred, labels)
                m_loss += loss.item()
                
                acc += (torch.max(pred, dim=-1).indices == labels).sum().item() / batch_size

            m_loss /= len(eval_loader)
            acc /= len(eval_loader)

            print(f" --------->  Evaluation: loss = {m_loss} || accuracy = {acc*100}%")
        # setting network back to training mode
        net.train()

print("Done!")
torch.save(net.state_dict(), checkpoint_file)