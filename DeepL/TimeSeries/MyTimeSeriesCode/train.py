optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# 假设 dataloader 是我们的数据加载器
for epoch in range(num_epochs):
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)  # label 需要是一个torch tensor，并且是长整型
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')
