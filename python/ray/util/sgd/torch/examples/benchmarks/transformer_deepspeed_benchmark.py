import argparse
import logging
import os
import time
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange
import torch.distributed as dist

from transformers import (HfArgumentParser, TrainingArguments, AutoTokenizer)
from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes

import ray
from ray import tune
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch.examples.transformers.utils import (
    evaluate, save_and_evaluate_checkpoints, load_and_cache_examples)
from ray.util.sgd.torch.examples.transformers.transformers_example import (set_seed,
                                                                           TransformerOperator,
                                                                           ModelArguments,
                                                                           DataProcessingArguments,
                                                                           RayArguments,
                                                                           model_creator,
                                                                           optimizer_creator)
from ray.util.sgd.torch.deepspeed.deepspeed_operator import deepspeed_cls
from ray.util.sgd.utils import (set_cuda_devices_list, get_benchmark_cls, summarize_mem_usage,
                                save_epoch_data, BATCH_SIZE)

logger = logging.getLogger(__name__)

@dataclass
class DeepSpeedArguments:
    use_deepspeed: bool = field(
        default=False,
        metadata={"help": "Use GPU memory optimizations from ZeRO"})


def data_creator(config):
    args = config["args"]
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    logger.info("tokenizer instantiation time: {}".format(time.time() - start))

    train_dataset = load_and_cache_examples(
        args, args.task_name, tokenizer, evaluate=False)
    train_dataset, _ = torch.utils.data.random_split(train_dataset,
                                                     [len(train_dataset)//2,
                                                      len(train_dataset)//2])
    train_sampler = RandomSampler(
        train_dataset) if not dist.is_initialized() else None
    return DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_gpu_train_batch_size)



def main():
    parser = HfArgumentParser((ModelArguments, DataProcessingArguments,
                               TrainingArguments, RayArguments, DeepSpeedArguments))
    all_args = parser.parse_args_into_dataclasses()
    model_args, dataprocessing_args, training_args, ray_args, deepspeed_args = all_args

    # For now, let's merge all the sets of args into one,
    # but soon, we'll keep distinct sets of args, with a
    # cleaner separation of concerns.
    args = argparse.Namespace(**vars(model_args), **vars(dataprocessing_args),
                              **vars(training_args), **vars(ray_args),
                              **vars(deepspeed_args))

    if args.use_deepspeed:
        Operator = deepspeed_cls(TransformerOperator)
    else:
        Operator = TransformerOperator

    Operator = get_benchmark_cls(Operator)

    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir)
            and args.do_train and not args.overwrite_output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome.".format(args.output_dir))

    use_gpu = not args.no_cuda

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    args.output_mode = output_modes[args.task_name]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    logger.info("Training/evaluation parameters %s", args)

    os.environ['CUDA_DEVICE_LIST'] = '1,2,3,'

    ray.init(address=args.address)
    # Training

    # to get directory for saving data very hacky
    def dummy_task(config, reporter):
        pass

    config = {'fp16': tune.grid_search([args.fp16]),
              'use_deepspeed': tune.grid_search([args.use_deepspeed]),
              'num_workers': tune.grid_search([args.num_workers]),
              'batch': tune.grid_search([args.per_gpu_train_batch_size])}
    analysis = ray.tune.run(dummy_task, config=config)
    directory = analysis._get_trial_paths()[0]
    print("saving data to", directory)

    trainer = TorchTrainer(
        model_creator=model_creator,
        data_creator=data_creator,
        optimizer_creator=optimizer_creator,
        training_operator_cls=Operator,
        use_fp16=args.fp16,
        apex_args={"opt_level": args.fp16_opt_level},
        num_workers=args.num_workers,
        use_gpu=use_gpu,
        use_tqdm=True,
        config={"args": args,
                # will be divided by number of workers before passed to data creator and operators
                BATCH_SIZE: args.per_gpu_train_batch_size * args.num_workers,
                # TODO decide how to deal with gradient accum for benchmark
                'num_workers': args.num_workers})

    args.device = torch.device("cuda" if use_gpu else "cpu")

    tokenizer = trainer.get_local_operator().tokenizer
    local_model = trainer.get_model()

    epochs_trained = 0
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch"
    )

    trainer.apply_all_workers(lambda: set_seed(args))
    if args.do_train:
        training_data = []
        for i in train_iterator:
            stats = trainer.train()
            print("Training stats:", stats)
            training_data.append(stats)
            save_epoch_data(stats, save_iteration=i, directory=directory)

    trainer.shutdown()
    print("-----------------------")
    print("success!")
    print(args)
    print("-----------------------")


if __name__ == "__main__":
    main()
