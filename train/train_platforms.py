import os

class TrainPlatform:
    def __init__(self, save_dir):
        pass

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from clearml import Task
        path, name = os.path.split(save_dir)
        self.task = Task.init(project_name='motion_diffusion',
                              task_name=name,
                              output_uri=path)
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir):
        pass


class WandbPlatform(TrainPlatform):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.run = None

    def _ensure_run(self, args=None):
        if self.run is not None:
            return self.run
        from mdm_core.utils.wandb_util import init_wandb_run
        self.run = init_wandb_run(save_dir=self.save_dir, args=args)
        return self.run

    def report_scalar(self, name, value, iteration, group_name=None):
        run = self._ensure_run()
        key = f'{group_name}/{name}' if group_name else name
        run.log({key: value, 'global_step': iteration}, step=iteration)

    def report_args(self, args, name):
        run = self._ensure_run(args=args)
        run.config.update(vars(args), allow_val_change=True)

    def close(self):
        if self.run is not None:
            self.run.finish()
            self.run = None


# Backward-compatible alias used by some existing args.json files.
WandBPlatform = WandbPlatform

