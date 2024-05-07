import hydra
import torch.nn
import torch.optim
from omegaconf import DictConfig


@hydra.main(config_path='', config_name='config_cube', version_base='1.1')
def main(cfg: DictConfig):  # noqa: C901

    device = cfg.device
    if device in ('', None):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    device = torch.device(device)


if __name__ == '__main__':
    main()
