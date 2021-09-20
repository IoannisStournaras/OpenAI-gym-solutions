import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from random import sample
from tqdm import tqdm
from typing import List, Tuple, Union

import gym
import numpy as np
from torch import nn, Tensor, cat
from PIL import Image, ImageOps

from reinforcement.models.cnn import ImageEnvironment2D
from reinforcement.settings import Settings

logger = logging.getLogger(__name__)

COMPLEX_ACTION_SPACE = {
    'Turn_left_hard_&_Accelerate_hard': (-1.0, 1.0, 0.0),
    'Turn_left_hard_&_Accelerate_soft': (-1.0, 0.5, 0.0),
    'Turn_left_soft_&_Accelerate_hard': (-0.5, 1.0, 0.0),
    'Turn_left_soft_&_Accelerate_soft': (-0.5, 0.5, 0.0),
    'Turn_left_hard_&_Break_hard': (-1.0, 0.0, 0.8),
    'Turn_left_hard_&_Break_soft': (-1.0, 0.0, 0.4),
    'Turn_left_soft_&_Break_hard': (-0.5, 0.0, 0.8),
    'Turn_left_soft_&_Break_soft': (-0.5, 0.0, 0.4),
    'Turn_right_hard_&_Accelerate_hard': (1.0, 1.0, 0.0),
    'Turn_right_hard_&_Accelerate_soft': (1.0, 0.5, 0.0),
    'Turn_right_soft_&_Accelerate_hard': (0.5, 1.0, 0.0),
    'Turn_right_soft_&_Accelerate_soft': (0.5, 0.5, 0.0),
    'Turn_right_hard_&_Break_hard': (1.0, 0.0, 0.8),
    'Turn_right_hard_&_Break_soft': (1.0, 0.0, 0.4),
    'Turn_right_soft_&_Break_hard': (0.5, 0.0, 0.8),
    'Turn_right_soft_&_Break_soft': (0.5, 0.0, 0.4),
    'Keep_straight_&_Accelerate_hard': (0.0, 1.0, 0.0),
    'Keep_straight_&_Accelerate_soft': (0.0, 0.5, 0.0),
    'Keep_straight_&_Break_hard': (0.0, 0.0, 0.8),
    'Keep_straight_&_Break_soft': (0.0, 0.0, 0.5),
    'Do_nothing': (0.0, 0.0, 0.0)
}

SIMPLE_ACTION_SPACE = {
    'Turn_left': (-1.0, 0.0, 0.0),
    'Turn_right': (1.0, 0.0, 0.0),
    'Break': (0.0, 0.0, 0.8),
    'Accelerate': (0.0, 1.0, 0.0),
    'Do-Nothing': (0.0, 0.0, 0.0),
}


class CarRacingAgent:
    def __init__(
        self,
        envinornment: str = 'CarRacing-v0',
        action_space: Union[str, List[Tuple[float, float, float]]] = 'simple',
        discount_factor: float = 0.99,
        epsilon: float = 0.9,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.99,
        learning_rate: float = 0.001,
        memory_size: int = 1000,
        n_stacked_frames: int = 4,
        output_dir: Union[Path, str] = None,
        input_dim: Tuple[int, int, int] = (1, 84, 84),
        model: nn.Module = None
    ):
        if isinstance(action_space, str):
            action_space = (
                list(COMPLEX_ACTION_SPACE.values()) if action_space.lower() == 'complex'
                else list(SIMPLE_ACTION_SPACE.values()) if action_space.lower() == 'simple'
                else None
            )

        if not action_space:
            raise ValueError('Action Space must be one of complex/simple or custom list')

        self.action_space = action_space

        self.n_stacked_frames = n_stacked_frames
        self.stacked_frames = None

        if n_stacked_frames > 1:
            input_dim = (n_stacked_frames, input_dim[1], input_dim[2])
            self.stacked_frames = deque(maxlen=n_stacked_frames)

        self.env = gym.make(envinornment)
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.model = model if model else ImageEnvironment2D(
                input_dim=input_dim, action_space=len(self.action_space), lr=learning_rate
            )
        self.target_model = model if model else ImageEnvironment2D(
                input_dim=input_dim, action_space=len(self.action_space), lr=learning_rate
            )

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        self.pretrained_dir = (
            output_dir
            or Settings.DATA_DIR.joinpath('pretrained', 'continuous')
        )
        self.pretrained_dir.mkdir(exist_ok=True, parents=True)

    def insert_to_memory(
        self,
        current_state: Union[np.ndarray, Tensor],
        action: Tuple[float, float, float],
        reward: float,
        next_state: Union[np.ndarray, Tensor],
        status: bool
    ):
        self.memory.append(
            (current_state, self.action_space.index(action), reward, next_state, status)
        )

    @staticmethod
    def _convert_rgb_to_grayscale(
        img: Union[Image.Image, np.ndarray],
        crop: bool = True
    ) -> np.ndarray:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, 'RGB')
        img = ImageOps.grayscale(img)
        img = np.array(img)
        if crop:
            img = img[:84, 12:]
        return np.expand_dims(img, axis=0)

    def act(self, state):
        if np.random.uniform(low=0, high=1) < self.epsilon:
            action_index = np.random.randint(0, len(self.action_space))
        else:
            action_index = self.model.predict(state).argmax().item()
        return self.action_space[action_index]

    def replay(self, batch_size: int = 64):
        train_sample = sample(self.memory, batch_size)
        train_state: List[np.ndarray] = []
        train_target: List[Tensor] = []
        for current_state, action_index, reward, next_state, status in train_sample:
            target: Tensor = self.model.predict(current_state)
            if status:
                target[0][action_index] = reward
            else:
                td: Tensor = self.target_model.predict(next_state)
                target[0][action_index] = reward + self.discount_factor * td.max().item()

            train_state.append(current_state)
            train_target.append(target)

        self.model.fit(np.array(train_state), cat(train_target), epochs=1)

    def _update_target_model_parameters(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _next_step(self, action: Tuple[float, float, float], skip_frames: int = 0):
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, status, _ = self.env.step(action)
            next_state = self._convert_rgb_to_grayscale(next_state)
            reward += r
            if status:
                break
        return next_state, reward, status

    def _get_stacked_state(self, state: np.ndarray, init: bool = False) -> np.ndarray:
        if init:
            self.stacked_frames.extend([state]*self.n_stacked_frames)
        else:
            self.stacked_frames.append(state)
        return np.concatenate(self.stacked_frames, axis=0)

    def train(
        self,
        episodes: int,
        skip_frames: int = 2,
        batch_size: int = 64,
        update_target_frequency: int = 5,
        max_negative_reward_tolerance: float = 0.1,
        max_continuous_negative_rewards: int = 50,
        enhance_accelarate_rewards: bool = True,
        render: bool = False,
        checkpoint_frequency: int = 100,
    ):
        stop_condition_names = ['Done', 'Continuous Negative Rewards', 'Max Negative Reward']
        for episode in tqdm(range(1, episodes+1)):
            total_reward = 0
            negative_reward_counter = 0
            n_time_frames = 1
            status = False

            current_state = self.env.reset()
            current_state = self._convert_rgb_to_grayscale(current_state)

            if self.n_stacked_frames > 1:
                current_state = self._get_stacked_state(current_state, init=True)

            while status is False:
                if render:
                    self.env.render()

                action = self.act(current_state)
                next_state, reward, status = self._next_step(action, skip_frames=skip_frames)
                if self.n_stacked_frames > 1:
                    next_state = self._get_stacked_state(next_state)

                if enhance_accelarate_rewards and action[1] >= 0:
                    reward *= 1.1
                total_reward += reward

                self.insert_to_memory(current_state, action, reward, next_state, status)
                current_state = next_state

                # If getting negative rewards terminate episode and start over
                negative_reward_counter = (
                    negative_reward_counter + 1 if n_time_frames > 100 and reward < 0
                    else 0
                )

                stop_conditions = [
                    status,
                    negative_reward_counter >= max_continuous_negative_rewards,
                    total_reward < -max_negative_reward_tolerance
                ]
                if any(stop_conditions):
                    condition = stop_condition_names[np.argmax(stop_conditions)]
                    logger.info(
                        f'Episode: {episode}/{episodes}, Time Frames: {n_time_frames}, '
                        f'Total Reward: {total_reward:.2}, Stop Condition: {condition}'
                    )
                    break

                if len(self.memory) > batch_size:
                    self.replay(batch_size)

                n_time_frames += 1

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if episode % update_target_frequency == 0:
                self._update_target_model_parameters()

            if episode % checkpoint_frequency == 0:
                filename = datetime.now().strftime(f'CarRacing_{episode}_%Y-%m-%dT%Hh%Mm%Ss.pth')
                self.save(filename)

        self.env.close()

    def play(self, render: bool = True):
        stored_epsilon = self.epsilon
        self.epsilon = 0
        total_reward = 0
        n_time_frames = 1
        status = False

        current_state = self.env.reset()
        current_state = self._convert_rgb_to_grayscale(current_state)

        if self.n_stacked_frames > 1:
            current_state = self._get_stacked_state(current_state, init=True)

        while status is False:
            if render:
                self.env.render()

            action = self.act(current_state)
            next_state, reward, status = self._next_step(action)
            if self.n_stacked_frames > 1:
                next_state = self._get_stacked_state(next_state)

            total_reward += reward
            current_state = next_state
            n_time_frames += 1

        self.epsilon = stored_epsilon
        self.env.close()
        return total_reward, n_time_frames

    def save(self, filename: Union[Path, str]):
        filename = self.pretrained_dir / filename
        self.model.save_model(filename)

    def load(self, filename: Union[Path, str]):
        filename = self.pretrained_dir / filename
        self.model.load_model(filename)