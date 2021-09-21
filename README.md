# Reinforcement Learning
Training Agents to play CarRacing and FrozenLake from OpenAI GYM. Implementing DQN algorithm with Pytorch to solve the
continuous environment problem. For solving the discrete environment, Q-Learning and SARSA are implemented

## Getting Started

* Clone repository:


1) ``git clone git@github.com:IoannisStournaras/OpenAI-gym-solutions.git`` using ssh
2) ``git clone https://key:token@github.com/IoannisStournaras/OpenAI-gym-solutions.git`` using https (key/token)


* Install dependencies:
1) Install Docker Engine. For install instructions, see:
    - [How to install Docker Engine on Windows](https://docs.docker.com/docker-for-windows/install/)
    - [How to install Docker Engine on Mac](https://docs.docker.com/docker-for-mac/install/)
    - [How to install Docker Engine on Fedora](https://docs.docker.com/engine/install/fedora/)
    - [How to install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
    - [Other distribution](https://docs.docker.com/engine/install/)
2) Install Docker Compose. 
    - For Mac/Windows users Docker Desktop includes Compose along with other Docker apps,
 so no need to install Compose separately. Please see install instructions above.
    - For Linux users, instructions available at [How to install Docker Compose](https://docs.docker.com/compose/install/)
    
For **Linux** users, you should add your user to the docker group. It is also recommended to configure docker to start on
boot. For instructions, please visit [Post-installation steps for Linux](https://docs.docker.com/engine/install/linux-postinstall/)
    
## Build and Run

You can build and run an environment by running the following steps: 
1) ``docker-compose build # builds docker image``
2)  ``docker-compose run --rm  discrete-play-q # runs one of the predefined services``

Other predefined choices are: `discrete-play-sarsa`, `continuous-play-simple`, `continuous-play-complex` and `custom`

You can execute any custom command by using the custom service. For example the command presented below trains a 
discrete agent using the Q-Learning algorithm on the FrozenLake environment:

``docker-compose run --rm  custom python3 reinforcement/main.py discrete train -e 5000 -m q_learning``




