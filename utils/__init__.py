import torch
import torch.distributed as dist
from collections import deque, defaultdict
import time
import datetime
import os

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        if d.numel() == 0:
            return 0.0
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        if d.numel() == 0:
            return 0.0
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return 0.0  # 安全返回0，避免除以零错误
        return self.total / self.count

    @property
    def max(self):
        if not self.deque:
            return 0.0
        return max(self.deque)

    @property
    def value(self):
        if not self.deque:
            return 0.0
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f"Value for {k} must be float or int, got {type(v)}"
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if name != "loss_spatial" or (name == "loss_spatial" and torch.tensor(list(meter.deque)).numel() != 0):
                loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            if meter.count > 0:
                loss_str.append(f"{name}: {meter.global_avg:.4f}")
            else:
                loss_str.append(f"{name}: 0.0000")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, dataset_len=None, epoch_info=None):
        if not header:
            header = ''
        if not dataset_len:
            dataset_len = len(iterable)
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(dataset_len))) + 'd'

        _msg = [
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            _msg.append('max mem: {memory:.0f}')
        _msg = self.delimiter.join(_msg)
        MB = 1024.0 * 1024.0
        iterable = iter(iterable)
        train_steps = dataset_len
        if epoch_info:
            start_epoch, end_epoch = epoch_info
            train_steps = (end_epoch - start_epoch) * dataset_len
        for i in range(train_steps):
            try:
                obj = next(iterable)
            except StopIteration:
                break
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if epoch_info:
                current_epoch = int(i / dataset_len) + start_epoch
                current_header = f'Train step: [{current_epoch}]'
            else:
                current_header = header
            log_msg = current_header + " " + _msg
            if (i % print_freq == 0) or (i == dataset_len - 1):
                eta_seconds = iter_time.global_avg * (dataset_len - i % dataset_len)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i % dataset_len, dataset_len, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i % dataset_len, dataset_len, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / dataset_len))

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def is_main_process():
    return get_rank() == 0

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return


# utils/__init__.py
class Meter:
    def __init__(self, window_size=20, fmt='{value:.4f}'):
        self.window_size = window_size
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.deque = deque(maxlen=self.window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.pop(0)
        self.total += value * n
        self.count += n

    @property
    def global_avg(self):
        if self.count == 0:
            return 0.0  # 安全返回0，避免除以零错误
        return self.total / self.count

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        if d.numel() == 0:
            return 0.0
        return d.median().item()

    @property
    def max(self):
        if not self.deque:
            return 0.0
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self):
        if not self.deque:
            return 0.0
        return self.deque[-1] if self.deque else 0.0

    def __str__(self):
        return self.fmt.format(value=self.global_avg)
