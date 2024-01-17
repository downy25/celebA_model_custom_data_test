import torch
import custum_data
model=torch.load('./models/celeba-cnn.ph')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
accuracy_test = 0

model.eval()
with torch.no_grad():
    for x_batch, y_batch in custum_data.test_data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch)[:, 0]
        is_correct = ((pred>=0.3).float() == y_batch).float()
        accuracy_test += is_correct.sum().cpu()

accuracy_test /= len(custum_data.test_data_loader.dataset)

print(f'테스트 정확도: {accuracy_test:.4f}')

pred = model(x_batch)[:, 0] * 100

fig = custum_data.plt.figure(figsize=(15, 7))
for j in range(0, 5):
    ax = fig.add_subplot(2, 5, j+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(x_batch[j].cpu().permute(1, 2, 0))
    if (pred[j]>=50):
        label = 'Smile'
    else:
        label = 'Not Smile'
    ax.text(
        0.5, -0.15,
        f'GT: {label:s}\nPr(Smile)={pred[j]:.0f}%',
        size=16,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes)

#plt.savefig('figures/figures-14_18.png', dpi=300)
custum_data.plt.show()
