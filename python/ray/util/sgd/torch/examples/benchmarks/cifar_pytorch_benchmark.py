import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import timeit
import humanize
import pandas as pd

from tqdm import trange

import nvidia_smi

import ray
from ray.util.sgd.torch import TorchTrainer
from ray.util.sgd.torch.resnet import ResNet18
from ray.util.sgd.utils import BATCH_SIZE
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.torch.deepspeed.deepspeed_operator import DeepSpeedOperator

GPU_USAGE_THRESHOLD = 10.0


def initialization_hook():
    # Need this for avoiding a connection restart issue on AWS.
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_LL_THRESHOLD"] = "0"

    # set the below if needed
    # print("NCCL DEBUG SET")
    # os.environ["NCCL_DEBUG"] = "INFO"


def cifar_creator(config):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])  # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = CIFAR10(
        root="~/data", train=True, download=True, transform=transform_train)
    validation_dataset = CIFAR10(
        root="~/data", train=False, download=False, transform=transform_test)

    if config["test_mode"]:
        train_dataset = Subset(train_dataset, list(range(64)))
        validation_dataset = Subset(validation_dataset, list(range(64)))

    train_loader = DataLoader(
        train_dataset, batch_size=config[BATCH_SIZE], num_workers=2)
    validation_loader = DataLoader(
        validation_dataset, batch_size=config[BATCH_SIZE], num_workers=2)
    return train_loader, validation_loader


def optimizer_creator(model, config):
    """Returns optimizer"""
    return torch.optim.SGD(
        model.parameters(),
        lr=config.get("lr", 0.1),
        momentum=config.get("momentum", 0.9))


def scheduler_creator(optimizer, config):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[150, 250, 350], gamma=0.1)


def get_gpu_mem_usage():
    if not args.use_gpu:
        return None
    devices = list(filter(lambda dev: bool(dev), os.environ.get("CUDA_VISIBLE_DEVICES").split(",")))  # TODO make this a function
    data = {
            #"allocated": 0,
            #"max_allocated": 0,
            #"reserved": 0,
            #"max_reserved": 0,
            "used": 0,
            "max_used": 0}
    for gpu in devices:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(int(gpu))
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        if res.gpu > GPU_USAGE_THRESHOLD:  # only take from GPUs in use
            mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            data["used"] += mem_res.used   # sum across all GPUs
            data["max_used"] += mem_res.used  # will be maxed across time at end


        # TODO doesn't work if on shared machine and CUDA_VISIBLE_DEVICES is not
        # TODO contiguous and starting at 0. e.g. 1,2,4 (someone is using 0 & 3)
        # TODO see https://github.com/pytorch/pytorch/issues/24463
        # TODO fix is to use the entire machine and set visible devices before
        # TODO running the job
        '''
        cuda_device = "cuda:" + str(gpu)
        data["allocated"] += torch.cuda.memory_allocated(cuda_device)
        data["max_allocated"] += torch.cuda.max_memory_allocated(cuda_device)
        data["reserved"] += torch.cuda.memory_reserved(cuda_device)
        data["max_reserved"] += torch.cuda.max_memory_reserved(cuda_device)
        '''
    df = pd.DataFrame(data, index=[0])
    return df


def summarize_mem_usage(df):
    if not args.use_gpu:
        return None
    import humanize  # makes memory usage human-readable
    data = []
    for col_name in df.columns:
        if "max" in col_name:
            data.append(humanize.naturalsize(df[col_name].max()))
        else:
            data.append(humanize.naturalsize(df[col_name].mean()))
    return pd.Series(data=data, index=df.mean().index)


if __name__ == "__main__":
    nvidia_smi.nvmlInit()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        required=False,
        type=str,
        help="the address to use for connecting to the Ray cluster")
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=1,
        help="Sets number of workers for training.")
    parser.add_argument(
        "--num-epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Enables GPU training")
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enables FP16 training with apex. Requires `use-gpu`.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Finish quickly for testing.")
    parser.add_argument(
        "--tune", action="store_true", default=False, help="Tune training")
    parser.add_argument(
        "--use-deepspeed",
        action="store_true",
        default=False,
        help="Use the GPU memory optimizations from ZeRO")

    args, _ = parser.parse_known_args()
    SuperClass = DeepSpeedOperator if args.use_deepspeed else TrainingOperator
    class Training(SuperClass):

        def setup(self, config):
            super(Training, self).setup(config)
            self.batch_size = config[BATCH_SIZE]

        def train_batch(self, batch, batch_info):
            def benchmark():
                return super(Training, self).train_batch(batch, batch_info)

            if self.global_step == 0:
                print("Running warmup...")
                timeit.timeit(benchmark, number=1)
                self.global_step += 1
            time = timeit.timeit(benchmark, number=1)
            img_sec = args.num_workers * self.batch_size / time
            return {"img_sec": img_sec}

    num_cpus = 4 if args.smoke_test else None
    ray.init(address=args.address, num_cpus=num_cpus, log_to_driver=True,
             include_webui=False)

    trainer1 = TorchTrainer(
        model_creator=ResNet18,
        data_creator=cifar_creator,
        optimizer_creator=optimizer_creator,
        loss_creator=nn.CrossEntropyLoss,
        scheduler_creator=scheduler_creator,
        initialization_hook=initialization_hook,
        num_workers=args.num_workers,
        config={
            "lr": 0.1,
            "test_mode": args.smoke_test,  # subset the data
            # this will be split across workers.
            BATCH_SIZE: 128 * args.num_workers
        },
        use_gpu=args.use_gpu,
        scheduler_step_freq="epoch",
        use_fp16=args.fp16,
        use_tqdm=True,
        training_operator_cls=Training)
    pbar = trange(args.num_epochs, unit="epoch")
    mem_usage = None
    for i in pbar:
        info = {"num_steps": 1} if args.smoke_test else {}
        info["epoch_idx"] = i
        info["num_epochs"] = args.num_epochs
        # Increase `max_retries` to turn on fault tolerance.
        train_stats = trainer1.train(max_retries=1, info=info)
        pbar.set_postfix(dict(throughput=train_stats["img_sec"]))
        if mem_usage is None:
            mem_usage = get_gpu_mem_usage()
        else:
            mem_usage = mem_usage.append(get_gpu_mem_usage(), ignore_index=True)
    print(trainer1.validate())
    trainer1.shutdown()
    print("success!")
    print("-----------------------")
    print("num workers:", args.num_workers)
    print("fp16:", args.fp16)
    print("ZeRO:", args.use_deepspeed)
    print("throughput: {} img/sec".format(train_stats['img_sec']))
    print("GPU Memory Usage:")
    print(summarize_mem_usage(mem_usage))
    print("----------------------")
