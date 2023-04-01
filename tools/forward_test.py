import numpy as np
import paddle
import torch
from reprod_log import ReprodLogger

from ppcls.arch.backbone.model_zoo.ibot import IBOT_ViT_small_patch16_224

def setup_seed(seed=10):
    import torch
    import os
    import numpy as np
    import random
    torch.manual_seed(seed)  # 为CPU设置随机种子
    paddle.seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        paddle.seed(seed)
        #os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    paddle.set_device("cpu")
    # load model
    # the model is save into ~/.cache/torch/hub/checkpoints/alexnet-owt-4df8aa71.pth

    # def logger
    reprod_logger = ReprodLogger()

    model = IBOT_ViT_small_patch16_224(pretrained=False)
    param_dict = paddle.load("/data1/linkaihao/reproduce/ibot/duiqi/student.pdparams")
    model.set_dict(param_dict)
    model.eval()
    setup_seed(10)

    torch_fake_data_list = []
    torch_fake_label_list = []
    torch_fake_mask_list = []

    paddle_fake_data_list = []
    paddle_fake_label_list = []
    paddle_fake_mask_list = []

    # np.random
    fake_data = np.random.rand(32, 3, 224, 224).astype(np.float32) - 0.5
    fake_mask = np.random.rand(32, 14, 14).astype(np.float32)
    fake_label = np.arange(1).astype(np.int64)
    # np.save("fake_data.npy", fake_data)
    # np.save("fake_mask.npy", fake_label)

    # fake_data = paddle.randn([32, 3, 224, 224])
    # fake_mask = paddle.randn([32, 14, 14]) > 0.5
    # torch_fake_data = torch.rand(32, 3, 224, 224)
    # torch_fake_mask = torch.randn(32, 14, 14) > 0.5

    torch_fake_data = torch.from_numpy(fake_data)
    torch_fake_mask = torch.from_numpy(fake_mask) > 0.5
    paddle_fake_data = paddle.to_tensor(fake_data)
    paddle_fake_mask = paddle.to_tensor(fake_mask) > 0.5



    # fake_mask = np.random.rand(1, 3, 14, 14) > 0.5
    fake_label = np.arange(1).astype(np.int64)
    for _ in range(0, 2):
        torch_fake_data_list.append(torch_fake_data)
        # fake_label_list.append(fake_label)
        torch_fake_mask_list.append(torch_fake_mask)
        paddle_fake_data_list.append(paddle_fake_data)
        # fake_label_list.append(fake_label)
        paddle_fake_mask_list.append(paddle_fake_mask)

    model(paddle_fake_data_list,mask=paddle_fake_mask_list)
    # # read or gen fake data
    # fake_data = np.load("../../fake_data/fake_data.npy")
    # fake_data = paddle.to_tensor(fake_data)
    # # forward
    # out = model(fake_data)
    # #
    # reprod_logger.add("logits", out.cpu().detach().numpy())
    # reprod_logger.save("forward_paddle.npy")