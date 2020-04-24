from ray.util.sgd.torch import TrainingOperator
import ray
from deepspeed.pt.deepspeed_zero_optimizer import FP16_DeepSpeedZeroOptimizer
import torch.distributed as dist
from deepspeed.pt.deepspeed_constants import TORCH_DISTRIBUTED_DEFAULT_PORT
from datetime import timedelta
from ray.util.sgd.torch.constants import NCCL_TIMEOUT_S


class DeepSpeedOperator(TrainingOperator):

    def setup(self, config):
        # only one process ray will not initialize process group
        # but deepspeed expects process group anyways
        if not dist.is_initialized():
            backend = "nccl"  # used by deepspeed but ray also supports gloo
            # Compute URL for initializing distributed PyTorch
            ip = ray.services.get_node_ip_address()
            port = TORCH_DISTRIBUTED_DEFAULT_PORT  # TODO something more robust here

            address = "tcp://{ip}:{port}".format(ip=ip, port=port)
            timeout = timedelta(seconds=NCCL_TIMEOUT_S)
            dist.init_process_group(backend=backend,
                                    init_method=address,
                                    rank=0,
                                    world_size=1,
                                    timeout=timeout)

        # wrap optimizers to be deepspeed optimizers
        self._optimizers = [FP16_DeepSpeedZeroOptimizer(op) for op in self._optimizers]