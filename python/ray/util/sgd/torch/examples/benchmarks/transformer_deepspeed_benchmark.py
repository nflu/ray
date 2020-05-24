import argparse
import logging
import json
import os
from dataclasses import dataclass, field

import torch
from tqdm import trange

from transformers import (HfArgumentParser, TrainingArguments)
from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes

import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch.examples.transformers.utils import (
    evaluate, save_and_evaluate_checkpoints)
from ray.util.sgd.torch.examples.transformers.transformers_example import (set_seed,
                                                                           TransformerOperator,
                                                                           ModelArguments,
                                                                           DataProcessingArguments,
                                                                           RayArguments,
                                                                           model_creator,
                                                                           data_creator,
                                                                           optimizer_creator)
from ray.util.sgd.torch.deepspeed.deepspeed_operator import deepspeed_cls

logger = logging.getLogger(__name__)

@dataclass
class DeepSpeedArguments:
    use_deepspeed: bool = field(
        default=False,
        metadata={"help": "Use GPU memory optimizations from ZeRO"})


def main():
    parser = HfArgumentParser((ModelArguments, DataProcessingArguments,
                               TrainingArguments, RayArguments, DeepSpeedArguments))
    all_args = parser.parse_args_into_dataclasses()
    model_args, dataprocessing_args, training_args, ray_args, deepspeed_args = all_args

    # For now, let's merge all the sets of args into one,
    # but soon, we'll keep distinct sets of args, with a
    # cleaner separation of concerns.
    args = argparse.Namespace(**vars(model_args), **vars(dataprocessing_args),
                              **vars(training_args), **vars(ray_args), **vars(deepspeed_args))

    if args.use_deepspeed:
        logger.info("Using DeepSpeed")
        operator = deepspeed_cls(TransformerOperator)
    else:
        operator = TransformerOperator


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
    ray.init(address=args.address)
    # Training

    trainer = TorchTrainer(
        model_creator=model_creator,
        data_creator=data_creator,
        optimizer_creator=optimizer_creator,
        training_operator_cls=operator,
        use_fp16=args.fp16,
        apex_args={"opt_level": args.fp16_opt_level},
        num_workers=args.num_workers,
        use_gpu=use_gpu,
        use_tqdm=True,
        config={"args": args})

    args.device = torch.device("cuda" if use_gpu else "cpu")

    tokenizer = trainer.get_local_operator().tokenizer
    local_model = trainer.get_model()

    epochs_trained = 0
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
    )

    trainer.apply_all_workers(lambda: set_seed(args))
    if args.do_train:
        for _ in train_iterator:
            stats = trainer.train()
            print("Training stats:", stats)
            logs = evaluate(args, local_model, tokenizer)
            print(json.dumps(logs))

    # Post-training validation
    save_and_evaluate_checkpoints(args, local_model, tokenizer)



if __name__ == "__main__":
    main()