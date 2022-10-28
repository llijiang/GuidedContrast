import logging
import os
import sys
import time
sys.path.append('../')


def create_logger(log_file):
    logger = logging.getLogger(__name__)
    level = logging.DEBUG
    logger.setLevel(level)

    log_format = '[%(asctime)s  %(levelname)s  %(filename)s  line %(lineno)d  %(process)d]  %(message)s'
    formatter = logging.Formatter(log_format)

    # build a stream handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # build a file handler
    logging.basicConfig(level=level, format=log_format, filename=log_file)

    return logger


def get_logger(cfg):
    if cfg.task == 'train':
        log_file = os.path.join(
            cfg.exp_path,
            f"train-{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.log"
        )
    elif cfg.task == 'test':
        log_file = os.path.join(
            cfg.result_dir,
            f"test-{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.log"
        )
    else:
        raise NotImplementedError
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = create_logger(log_file)
    logger.info('************************ Start Logging ************************')
    return logger