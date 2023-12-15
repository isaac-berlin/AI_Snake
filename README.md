<div align='center'>
<h1>Snake Game AI: A Comparative Study of A-Star Algorithm and Deep-Q Neural Network Approaches</h1>
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
<img src="https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white" />
<img src="https://img.shields.io/badge/Pygame-7A1FA2?style=for-the-badge&logo=python&logoColor=blue" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white" />
</div>



### About Our Study

This codebase is part of an in-depth research study examining deterministic and non-deterministic approaches for training automated agents in playing the Snake game. Specifically, our study compares the performance of the A* algorithm with a deep-q neural network.

### Detailed Research Findings

For detailed insights, analysis, and comprehensive results, explore the [full research paper](Results/Results.pdf) available in the "[Results](Results)" folder of this repository. The research paper provides extensive details on our experimental methodologies, findings, and conclusions drawn from the comparison between the A* algorithm and the deep-q neural network in playing the Snake game.

### Running the Code
To run the code, you will need to install the following dependencies as specified in the [requirements.txt](requirements.txt) file:

The main dependencies are:
- Python (3.11.5)
- Pygame
- PyTorch
- MatPlotLib.pyplot
- IPython

To run the Deep-Q Neural Network you need to run the ```agent.py``` file in the [Deep-Q](Deep-Q) directory. To run the A-Star Algorithm you need to run the ```game.py``` file in the [A-Star](A-Star) directory. 
If you want to play the game yourself without any AI, you can run the ```snake_game_human.py``` file in the root directory.

### The Snake Game Implementation Credit

The Snake game implementation found here is originally credited to [Patrick Loeber](https://github.com/patrickloeber). You can explore the original implementation [here](https://github.com/patrickloeber/python-fun/tree/master/snake-pygame) or watch a detailed explanation in [this video](https://www.youtube.com/watch?v=L8ypSXwyBds).