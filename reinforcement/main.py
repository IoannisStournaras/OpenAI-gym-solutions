import logging
from datetime import datetime

from reinforcement.settings import (
    configure_logger,
    prepare_options,
    Settings,
)
from reinforcement.agents import AgentType, DiscreteAgent, CarRacingAgent

logger = logging.getLogger(__name__)


def get_agent(args) -> AgentType:
    if args.agent == 'continuous':
        agent = CarRacingAgent(
            action_space=args.method,
            n_stacked_frames=args.stacked_frames,
            epsilon=args.epsilon,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor
        )
    elif args.agent == 'continuous':
        agent = DiscreteAgent(
            method=args.method,
            discount_factor=args.discount_factor,
            epsilon=args.epsilon,
            learning_rate=args.learning_rate,
        )
    else:
        raise RuntimeError

    if args.pretrained:
        agent.load(args.pretrained)

    return agent


def start():
    try:
        args = prepare_options()
        logfile = datetime.today().strftime('%Y-%m-%dT%Hh%Mm%Ss.log')
        configure_logger(args.loglevel, logfile=Settings.LOG_DIR.joinpath(logfile))

        agent = get_agent(args)
        if args.command == 'train':
            logger.info('Starting Training Process')
            agent.train(episodes=args.episodes)
        elif args.command == 'play':
            if args.pretrained is None:
                logger.warning('No Pretrained Model specified. Play will use Random Weights')
            agent.play(render=args.render)
        else:
            raise RuntimeError

    except Exception as error:
        logger.error(error)
        raise


if __name__ == '__main__':
    start()
