"""Code."""
import logging.config
import time
import torch

def singleton(cls):
    """Run singleton method."""
    # code.
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return get_instance()


@singleton
class Logger:
    """Define Class Logger."""

    def __init__(self):
        """Run __init__ method."""
        # code.
        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(process)d [%(filename)s:%(lineno)d] %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger("root")

    def set_logger(self, env="terminal"):
        """Run set_logger method."""
        # code.
        """
        default evn is terminal
        when we connect our agent with web service, we need to automatic output all logger into one file, pls set evn='web'
        :param env:
        :return:
        """
        if env == "terminal":
            logging.basicConfig(
                format="%(asctime)s %(levelname)s %(process)d [%(filename)s:%(lineno)d] %(message)s",
                level=logging.INFO,
            )
            self.logger = logging.getLogger("root")
        else:
            self.logger = logging.getLogger("root")
            self.logger.setLevel(logging.INFO)

            BASIC_FORMAT = "%(asctime)s %(levelname)s %(process)d [%(filename)s:%(lineno)d] %(message)s"
            formatter = logging.Formatter(BASIC_FORMAT)

            # stdout
            chlr = logging.StreamHandler()
            chlr.setLevel(logging.INFO)
            chlr.setFormatter(formatter)
            # log file
            file_name = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + ".log"
            fhlr = logging.FileHandler(file_name)
            fhlr.setLevel(logging.INFO)
            fhlr.setFormatter(formatter)

            self.logger.addHandler(chlr)
            self.logger.addHandler(fhlr)


rank_0_logger = Logger.logger


def print_rank_0(message):
    """Run print_rank_0 method."""
    # code.
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            rank_0_logger.info(message)
    else:
        rank_0_logger.info(message)