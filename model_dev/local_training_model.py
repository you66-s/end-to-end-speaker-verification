import time
from model_blocks import SpeakerEmbeddingCNN
from triplet_dataset import TripletDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as T
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
train_dataset = TripletDataset(csv_file=r"data\triplet_dataset\train_triplets.csv",transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

model = SpeakerEmbeddingCNN(embedding_dim=64).to(device)
criterion = nn.TripletMarginLoss(margin= 0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
start = time.time()
for epoch in range(1):
    print("start training...")
    for anchor, positive, negative in train_loader:
        print("batch...")
        start_batch = time.time()
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        loss = criterion(anchor_emb, positive_emb, negative_emb)

        # 3) Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end_batch = time.time()
        print(f"Batch time: {end_batch - start_batch:.2f} seconds")
    print(f"Epoch {epoch} — Loss={loss.item():.4f}")

end = time.time()
print(f"\nTraining time: {end - start:.2f} seconds")