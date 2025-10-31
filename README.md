# Green Pong: Playing Pong Sustainably

This repository contains the code and materials for the research essay "Green Pong: Playing Pong Sustainably" by Daniel Pascal Hefti. The project investigates how reinforcement learning agents can be trained to play Pong in a way that minimizes energy consumption, using CodeCarbon to track environmental impact.

## Repository Structure

- `dqn.py`  
	Main script implementing a Deep Q-Network (DQN) agent for the Atari Pong environment. Includes training and evaluation routines, with energy tracking via CodeCarbon.
- `requirements.txt`  
	Python dependencies for running the experiments.
- `dockerfile`  
	Containerized environment for reproducible research.
- `.codecarbon.config`  
	Configuration for CodeCarbon energy tracking.
- `LICENSE`  
	Project license information.
- `README.md`  
	Project overview and instructions.

## Main Scripts

- `dqn.py`  
	- `train_dqn()`: Trains a DQN agent on Atari Pong, logging energy usage and saving the trained model.
	- `evaluate_dqn()`: Evaluates the trained agent, reporting performance and energy consumption.

## Getting Started

1. **Clone the repository.**
2. **Install dependencies:**
	 ```sh
	 pip install -r requirements.txt
	 ```
3. **Run in Docker (recommended for reproducibility):**
	 ```sh
	 docker build -t green-pong .
	 docker run --rm green-pong
	 ```
4. **Or run locally:**
	 ```sh
	 python dqn.py
	 ```

## Project Purpose

The goal of this project is to evaluate and promote sustainable AI practices by training reinforcement learning agents to play Pong efficiently. The repository provides tools for automatic evaluation, energy tracking, and reproducible experiments.

---
For more details, see the research essay or contact the author.
