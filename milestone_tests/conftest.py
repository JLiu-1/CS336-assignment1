import torch


def pytest_sessionstart(session):
    # 微型 CPU 测试避免大量线程调度开销；不影响单独运行训练命令。
    torch.set_num_threads(1)
