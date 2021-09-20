import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

DISCRETE_ENVIRONMENTS = ['FrozenLake8x8-v1']
CONTINUOUS_ENVIRONMENTS = ['CarRacing-v0']
DISCRETE_METHODS = ['q_learning', 'sarsa']
CONTINUOUS_METHODS = ['complex', 'simple']


class _Settings:
    def __init__(self):
        self.default = Path(__file__).parent.absolute().parent / 'data'

    @staticmethod
    def _get_value(name: str, silent: bool = False) -> str:
        name = name.upper()
        value = os.environ.get(name)

        if not value and silent is False:
            raise RuntimeError(f'"{name}" is not set!')

        return value

    @staticmethod
    def create(path: Path) -> Path:
        path.mkdir(exist_ok=True, parents=True)
        return path

    @property
    def DATA_DIR(self) -> Path:
        path = Path(
            self._get_value('DATA_PATH', silent=True)
            or str(self.default)
        )
        return self.create(path)

    @property
    def LOG_DIR(self) -> Path:
        path = Path(
            self._get_value('LOG_DIR', silent=True)
            or self.DATA_DIR.joinpath('logs', datetime.today().strftime('%Y-%m-%d'))
        )
        return self.create(path)


def configure_logger(level: str = 'INFO', verbose: bool = True, logfile: Path = None):
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName(level.upper()))
    formatter = logging.Formatter(
        '%(asctime)s:%(name)s:%(levelname)s:%(module)s:%(funcName)s:%(message)s'
    )

    if verbose:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if logfile:
        hf = logging.FileHandler(
            filename=logfile.absolute().as_posix(),
            encoding='utf-8'
        )
        hf.setFormatter(formatter)
        logger.addHandler(hf)


def prepare_options():
    def check_filename(file: str):
        if file.split('.')[-1] not in ['pth', 'npz']:
            raise RuntimeError('Filename must be either of type .npz or .pth')

    parser = argparse.ArgumentParser(
        prog='Reinforcement',
        description="Reinforcement Learning Agent Training and Execution"
    )
    subparsers = parser.add_subparsers(help='sub-command help', dest='agent')

    # Discrete Agent arguments
    parser_discrete = subparsers.add_parser('discrete', help='discrete help')
    parser_discrete.add_argument(
        "-n", dest='environment', nargs="?", default='FrozenLake8x8-v1', type=str,
        choices=DISCRETE_ENVIRONMENTS, help='Specifies OpenAI Gym Environment to use'
    )
    parser_discrete.add_argument(
        "-m", dest='method', nargs="?", default='q_learning', type=str,
        choices=DISCRETE_METHODS, help='Specifies method to use for training the Agent'
    )

    # Continuous Agent arguments
    parser_continuous = subparsers.add_parser('continuous', help='continuous help')
    parser_continuous.add_argument(
        "-n", dest='environment', nargs="?", default='CarRacing-v0', type=str,
        choices=CONTINUOUS_ENVIRONMENTS, help='Specifies OpenAI Gym Environment to use'
    )
    parser_continuous.add_argument(
        "-m", dest='method', nargs="?", default='simple', type=str,
        choices=CONTINUOUS_METHODS, help='Specifies method to use for training the Agent'
    )
    parser_continuous.add_argument(
        '--stackframes', nargs="?", default=4, type=int,
        help='Defines the number of frames to stack on continuous Image Agents',
    )
    parser_continuous.add_argument(
        "--render", default=False, action='store_true',
        help='Use this option to activate rendering continuous environments'
    )

    # Common arguments for both Agents
    parser.add_argument(
        'command', type=str.lower, choices=['train', 'play'],
        help='Specifies which action to perform for an Agent Train/Play'
    )
    parser.add_argument(
        '-e', dest='episodes', nargs="?", default=100, type=int,
        help='Specifies number of episodes for the Agent to train'
    )
    parser.add_argument(
        "-d", dest='discount_factor', type=float, default=0.99,
        help="Determines how much the agents cares about immediate vs future rewards"
    )
    parser.add_argument(
        "-l", dest='learning_rate', nargs="?", default=0.1, type=float,
        help="Determines the step size at each iteration while minimizing a loss function"
    )
    parser.add_argument(
        '--epsilon', nargs="?", default=0.1, type=float,
        help="Used to select action from the Q values. Exploration vs Exploitation"
    )
    parser.add_argument(
        '--pretrained', nargs="?", default=None, type=check_filename,
        help="Used to select action from the Q values. Exploration vs Exploitation"
    )
    parser.add_argument(
        '--loglevel', nargs="?", default='INFO', type=str.upper, help='Defines Log Level',
        choices=logging._nameToLevel.keys(),
    )

    return parser.parse_args()


Settings = _Settings()
