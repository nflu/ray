import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import timeit
import numpy as np
from datetime import datetime


from tqdm import trange, tqdm

import ray
from ray.util.sgd.torch import TorchTrainer
from ray.util.sgd.torch.resnet import ResNet50, ResNet101, ResNet152
from ray.util.sgd.utils import BATCH_SIZE, get_gpu_mem_usage, summarize_mem_usage, set_cuda_devices_list
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.torch.deepspeed.deepspeed_operator import deepspeed_cls
from ray.util.sgd.torch.constants import (SCHEDULER_STEP_EPOCH, NUM_STEPS,
                                          SCHEDULER_STEP_BATCH, SCHEDULER_STEP)

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


if __name__ == "__main__":
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
        "--num-cpus",
        type=int,
        default=None,
        help="Sets number of cpus for training.")
    parser.add_argument(
        "--num-epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Enables GPU training")
    parser.add_argument(
        "--model",
        type=int,
        default=50,
        help="resnet model layers")
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
    set_cuda_devices_list(args.num_workers)
    class Training(TrainingOperator):

        def setup(self, config):
            super(Training, self).setup(config)
            self.batch_size = config[BATCH_SIZE]
            self.gpu_stats = None

        def train_batch(self, batch, batch_info):
            def benchmark():
                return super(Training, self).train_batch(batch, batch_info)

            if self.global_step == 0:
                print("Running warmup...")
                timeit.timeit(benchmark, number=1)
                self.global_step += 1

            time = timeit.timeit(benchmark, number=1)
            img_sec = args.num_workers * self.batch_size / time

            self.gpu_stats = get_gpu_mem_usage(data=self.gpu_stats)
            if 'img_sec' not in self.gpu_stats:
                self.gpu_stats['img_sec'] = []
            self.gpu_stats["img_sec"].append(img_sec)
            return self.gpu_stats

        def train_epoch(self, iterator, info):
            if self.use_tqdm and self.world_rank == 0:
                desc = ""
                if info is not None and "epoch_idx" in info:
                    if "num_epochs" in info:
                        desc = "{}/{}e".format(info["epoch_idx"] + 1,
                                               info["num_epochs"])
                    else:
                        desc = "{}e".format(info["epoch_idx"] + 1)
                _progress_bar = tqdm(
                    total=info[NUM_STEPS] or len(self.train_loader),
                    desc=desc,
                    unit="batch",
                    leave=False)

            self.model.train()
            for batch_idx, batch in enumerate(iterator):
                batch_info = {
                    "batch_idx": batch_idx,
                    "global_step": self.global_step
                }
                batch_info.update(info)
                metrics = self.train_batch(batch, batch_info=batch_info)

                if self.use_tqdm and self.world_rank == 0:
                    _progress_bar.n = batch_idx + 1
                    postfix = {}
                    if "train_loss" in metrics:
                        postfix.update(loss=metrics["train_loss"])
                    _progress_bar.set_postfix(postfix)

                if self.scheduler and batch_info.get(
                    SCHEDULER_STEP) == SCHEDULER_STEP_BATCH:
                    self.scheduler.step()

                self.global_step += 1

            if self.scheduler and info.get(SCHEDULER_STEP) == SCHEDULER_STEP_EPOCH:
                self.scheduler.step()

            return metrics

    print("-----------------------")
    print("starting")
    print("num workers:", args.num_workers)
    print("fp16:", args.fp16)
    print("ZeRO:", args.use_deepspeed)
    print("Resnet:", args.model)
    print("-----------------------")

    train_op_cls = deepspeed_cls(Training) if args.use_deepspeed else Training

    model_creator = {50:ResNet50, 101:ResNet101, 152:ResNet152}[args.model]

    ray.init(address=args.address, num_cpus=args.num_cpus, log_to_driver=True,
             include_webui=False)

    trainer1 = TorchTrainer(
        model_creator=model_creator,
        data_creator=cifar_creator,
        optimizer_creator=optimizer_creator,
        loss_creator=nn.CrossEntropyLoss,
        scheduler_creator=scheduler_creator,
        initialization_hook=initialization_hook,
        num_workers=args.num_workers,
        config={
            "lr": 0.1,
            # this will be split across workers.
            BATCH_SIZE: 128 * args.num_workers,
            "test_mode":False
        },
        use_gpu=args.use_gpu,
        scheduler_step_freq="epoch",
        use_fp16=args.fp16,
        use_tqdm=True,
        training_operator_cls=train_op_cls)
    pbar = trange(args.num_epochs, unit="epoch")
    data = None
    for i in pbar:
        info = {}
        info["epoch_idx"] = i
        info["num_epochs"] = args.num_epochs
        # Increase `max_retries` to turn on fault tolerance.
        data = trainer1.train(max_retries=1, info=info,
                                    reduce_results=False)
    trainer1.shutdown()

    # TODO use something based on experiment dir
    now = datetime.now()
    os.chdir(os.path.expanduser("~"))
    experiment_name = 'num_workers_' + str(args.num_workers) + 'fp16_' + str(args.fp16) + \
                        'ZeRO_' + str(args.use_deepspeed) + 'Resnet_' + str(args.model) + \
                            now.strftime("%m-%d-%Y-%H_%M_%S")
    dirname = os.path.join('ray_results', experiment_name)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    os.chdir(dirname)
    print(os.getcwd())
    print("-----------------------")
    print("success!")
    print("num workers:", args.num_workers)
    print("fp16:", args.fp16)
    print("ZeRO:", args.use_deepspeed)
    print("Resnet:", args.model)
    summarize_mem_usage(data, display=True, save='data')
    print("-----------------------")

