# rl-graph-cube

Rubik's cube graph solver. RL based.

## Development

1. Create VSCode project with `code .`
2. Install poery dependencies and add environment for python linting. Use `poetry config virtualenvs.in-project true` for creation env folder inside project. Then `poetry install --with dev`.
3. Inside container use:

    - `pytest -v -s -x` for all tests
    - use `python -m IPython` to check code

## Important thinks

### Deep cube

- [DeepCube](https://openreview.net/pdf?id=Hyfn2jCcKm)
- [DeepCubeA](https://cse.sc.edu/~foresta/assets/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf)
- [DeepCubeA github](https://github.com/forestagostinelli/DeepCubeA)
- [DeepCubeA UI](https://deepcube.igb.uci.edu/)

### Gym

- [Gym Rubiks Cube](https://github.com/mgroling/GymRubiksCube)
- [RubiksCubeGym](https://github.com/DoubleGremlin181/RubiksCubeGym/)
- [gym-Rubiks-Cube](https://github.com/RobinChiu/gym-Rubiks-Cube)

### Other

- [Santa23 Template](https://www.kaggle.com/code/alexandervc/santa23-template)
- [Solver cube reddit](https://www.reddit.com/r/Damnthatsinteresting/comments/yzq15g/now_the_legendary_rubiks_cube_is_easy_to/)
- [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#reinforcement-learning-dqn-tutorial)
- [Multi-agent Reinforcement Learning with Graph Q-Networks for Antenna Tuning WP](https://arxiv.org/pdf/2302.01199.pdf)
- [GQSAT](https://github.com/NVIDIA/GraphQSat)
