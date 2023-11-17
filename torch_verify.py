import torch

# GPU 사용 가능 -> True, GPU 사용 불가 -> False
print(torch.cuda.is_available())

# 사용 가능 GPU 개수 체크
print(torch.cuda.device_count()) # 3