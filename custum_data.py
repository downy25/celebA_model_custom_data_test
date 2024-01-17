from torchvision import datasets, transforms
import torch
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt

# 이미지 전처리 및 변환 정의
transform = transforms.Compose([
    transforms.CenterCrop([178, 178]),
    transforms.Resize((64, 64)),  # 필요한 크기로 조절
    transforms.ToTensor(),
])

# CelebA 데이터셋 로드 (여기서는 root 디렉토리는 사용자 정의 데이터셋이 있는 디렉토리로 변경해야 합니다.)
celeba_dataset = torchvision.datasets.ImageFolder(root='./Smile_classification/celeba/img_align_celeba',transform=transform)
print('test 데이터 갯수: ',len(celeba_dataset))

# fig = plt.figure(figsize=(15, 7))
# for i in range(0,10):
#     ax = fig.add_subplot(2, 5, i+1)
#     ax.set_xticks([]); ax.set_yticks([])
#     plt.imshow(celeba_dataset[i][0].permute(1,2,0))
# plt.show()

# DataLoader 생성
batch_size = 30
test_data_loader = torch.utils.data.DataLoader(celeba_dataset, batch_size=batch_size, shuffle=False)
