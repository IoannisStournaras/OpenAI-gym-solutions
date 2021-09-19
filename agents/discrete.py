from typing import Union
from pathlib import Path

import gym
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DiscreteAgent:
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        discount_factor: float = 0.99,
        env_name: str = 'FrozenLake8x8-v1',
        init_value: int = 0
    ):
        if not (0 <= epsilon <= 1):
            raise ValueError('Epsilon must be in the range [0, 1]')

        if not (0 <= discount_factor <= 1):
            raise ValueError('Epsilon must be in the range [0, 1)')

        self.epsilon = epsilon
        self.gamma = discount_factor
        self.alpha = learning_rate
        self.init_value = init_value
        self.env = gym.make(env_name)
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))

        if init_value:
            self.q_table[:] = init_value

    def act(self, state: int, epsilon: float = None) -> int:
        if epsilon is None:
            epsilon = self.epsilon

        action_distribution = self.q_table[state]
        # Greedy policy always chooses best action on each state
        max_value = action_distribution.max()
        # Initially when all values are zeroes (or generally equal) argmax would always
        # return the first element with max value and thus the sample space would be
        # limited to only a subset of available actions. To account for this behaviour
        action_to_take = np.random.choice(np.where(action_distribution == max_value)[0])

        if np.random.uniform(low=0, high=1) < epsilon:
            action_to_take = self.env.action_space.sample()
        return action_to_take

    def learn(
        self,
        current_state: int,
        action: int,
        reward: Union[float, int],
        next_state: int,
        status: bool,
        epsilon: float,
    ):
        next_action = None
        prior_state_action_value = self.q_table[current_state, action]
        if status is False:
            next_action = self.act(next_state, epsilon=epsilon)
            td_target = reward + self.gamma * self.q_table[next_state, next_action]
        else:
            td_target = reward

        td_delta = td_target - prior_state_action_value
        self.q_table[current_state, action] += self.alpha * td_delta
        return next_action

    def reset(self):
        init_state = self.env.reset()
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        return init_state

    def train(
        self,
        n_episodes: int,
        method: str = 'q_learning',
        output_path: Union[Path, str] = None,
    ):
        if method.lower() not in ['q_learning', 'sarsa']:
            raise ValueError('Method must be one of q_learning/sarsa')

        logger.info('Starting training process with {method}')
        goals = 0
        holes = 0
        for episode in tqdm(range(1, n_episodes+1), unit='Episode'):
            status = False
            current_state = self.env.reset()
            action = self.act(state=current_state, epsilon=self.epsilon)
            steps = 0

            while status is False:
                next_state, reward, status, info = self.env.step(action)
                next_action = self.learn(
                    current_state=current_state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    status=status,
                    epsilon=0 if method == 'q_learning' else self.epsilon
                )

                current_state = next_state
                action = (
                    next_action if method == 'sarsa'
                    else self.act(state=next_state, epsilon=self.epsilon)
                )
                steps += 1

                if status:
                    # Evaluating training
                    result = 'Goal' if reward == 1 else 'Hole'
                    if reward == 1:
                        goals += 1
                    elif reward == 0:
                        holes += 1
                    logger.info(f'Episode {episode} resulted in {result} in {steps} steps')

            if episode % 500 == 0:
                print(f'Episode {episode} Total Reward {goals}')
                print(f'Training Average reward per episode {goals / episode}')

        if output_path:
            self.save(output_path)
        return goals, holes

    def play(self):
        steps = 0
        status = False
        current_state = self.env.reset()
        result = 'Steps Exceeded'
        while status is False:
            action = self.act(state=current_state, epsilon=0)
            current_state, reward, status, _ = self.env.step(action)
            if status:
                result = 'Goal' if reward == 1 else 'Hole'
            steps += 1
        return result, steps

    def save(self, path: Union[Path, str]):
        with open(path, 'wb') as f:
            np.save(f, self.q_table)

    def load(self, path: Union[Path, str]):
        with open(path, 'rb') as f:
            self.q_table = np.load(f)
