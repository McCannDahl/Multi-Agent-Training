import os

import imageio
import numpy as np
import torch
from agilerl.algorithms.dqn import DQN
import random
from PIL import Image, ImageDraw, ImageFont

from pettingzoo.classic import connect_four_v3

class Opponent:
   """Connect 4 opponent to train and/or evaluate against.

   :param env: Environment to learn in
   :type env: PettingZoo-style environment
   :param difficulty: Difficulty level of opponent, 'random', 'weak' or 'strong'
   :type difficulty: str
   """

   def __init__(self, env, difficulty):
      self.env = env.env
      self.difficulty = difficulty
      if self.difficulty == "random":
         self.getAction = self.random_opponent
      elif self.difficulty == "weak":
         self.getAction = self.weak_rule_based_opponent
      else:
         self.getAction = self.strong_rule_based_opponent
      self.num_cols = 7
      self.num_rows = 6
      self.length = 4
      self.top = [0] * self.num_cols

   def update_top(self):
      """Updates self.top, a list which tracks the row on top of the highest piece in each column."""
      board = np.array(self.env.env.board).reshape(self.num_rows, self.num_cols)
      non_zeros = np.where(board != 0)
      rows, cols = non_zeros
      top = np.zeros(board.shape[1], dtype=int)
      for col in range(board.shape[1]):
         column_pieces = rows[cols == col]
         if len(column_pieces) > 0:
               top[col] = np.min(column_pieces) - 1
         else:
               top[col] = 5
      full_columns = np.all(board != 0, axis=0)
      top[full_columns] = 6
      self.top = top

   def random_opponent(self, action_mask, last_opp_move=None, block_vert_coef=1):
      """Takes move for random opponent. If the lesson aims to randomly block vertical wins with a higher probability, this is done here too.

      :param action_mask: Mask of legal actions: 1=legal, 0=illegal
      :type action_mask: List
      :param last_opp_move: Most recent action taken by agent against this opponent
      :type last_opp_move: int
      :param block_vert_coef: How many times more likely to block vertically
      :type block_vert_coef: float
      """
      if last_opp_move is not None:
         action_mask[last_opp_move] *= block_vert_coef
      action = random.choices(list(range(self.num_cols)), action_mask)[0]
      return action

   def weak_rule_based_opponent(self, player):
      """Takes move for weak rule-based opponent.

      :param player: Player who we are checking, 0 or 1
      :type player: int
      """
      self.update_top()
      max_length = -1
      best_actions = []
      for action in range(self.num_cols):
         possible, reward, ended, lengths = self.outcome(
               action, player, return_length=True
         )
         if possible and lengths.sum() > max_length:
               best_actions = []
               max_length = lengths.sum()
         if possible and lengths.sum() == max_length:
               best_actions.append(action)
      best_action = random.choice(best_actions)
      return best_action

   def strong_rule_based_opponent(self, player):
      """Takes move for strong rule-based opponent.

      :param player: Player who we are checking, 0 or 1
      :type player: int
      """
      self.update_top()

      winning_actions = []
      for action in range(self.num_cols):
         possible, reward, ended = self.outcome(action, player)
         if possible and ended:
               winning_actions.append(action)
      if len(winning_actions) > 0:
         winning_action = random.choice(winning_actions)
         return winning_action

      opp = 1 if player == 0 else 0
      loss_avoiding_actions = []
      for action in range(self.num_cols):
         possible, reward, ended = self.outcome(action, opp)
         if possible and ended:
               loss_avoiding_actions.append(action)
      if len(loss_avoiding_actions) > 0:
         loss_avoiding_action = random.choice(loss_avoiding_actions)
         return loss_avoiding_action

      return self.weak_rule_based_opponent(player)  # take best possible move

   def outcome(self, action, player, return_length=False):
      """Takes move for weak rule-based opponent.

      :param action: Action to take in environment
      :type action: int
      :param player: Player who we are checking, 0 or 1
      :type player: int
      :param return_length: Return length of outcomes, defaults to False
      :type player: bool, optional
      """
      if not (self.top[action] < self.num_rows):  # action column is full
         return (False, None, None) + ((None,) if return_length else ())

      row, col = self.top[action], action
      piece = player + 1

      # down, up, left, right, down-left, up-right, down-right, up-left,
      directions = np.array(
         [
               [[-1, 0], [1, 0]],
               [[0, -1], [0, 1]],
               [[-1, -1], [1, 1]],
               [[-1, 1], [1, -1]],
         ]
      )  # |4x2x2|

      positions = np.array([row, col]).reshape(1, 1, 1, -1) + np.expand_dims(
         directions, -2
      ) * np.arange(1, self.length).reshape(
         1, 1, -1, 1
      )  # |4x2x3x2|
      valid_positions = np.logical_and(
         np.logical_and(
               positions[:, :, :, 0] >= 0, positions[:, :, :, 0] < self.num_rows
         ),
         np.logical_and(
               positions[:, :, :, 1] >= 0, positions[:, :, :, 1] < self.num_cols
         ),
      )  # |4x2x3|
      d0 = np.where(valid_positions, positions[:, :, :, 0], 0)
      d1 = np.where(valid_positions, positions[:, :, :, 1], 0)
      board = np.array(self.env.env.board).reshape(self.num_rows, self.num_cols)
      board_values = np.where(valid_positions, board[d0, d1], 0)
      a = (board_values == piece).astype(int)
      b = np.concatenate(
         (a, np.zeros_like(a[:, :, :1])), axis=-1
      )  # padding with zeros to compute length
      lengths = np.argmin(b, -1)

      ended = False
      # check if winnable in any direction
      for both_dir in board_values:
         # |2x3|
         line = np.concatenate((both_dir[0][::-1], [piece], both_dir[1]))
         if "".join(map(str, [piece] * self.length)) in "".join(map(str, line)):
               ended = True
               break

      # ended = np.any(np.greater_equal(np.sum(lengths, 1), self.length - 1))
      draw = True
      for c, v in enumerate(self.top):
         draw &= (v == self.num_rows) if c != col else (v == (self.num_rows - 1))
      ended |= draw
      reward = (-1) ** (player) if ended and not draw else 0

      return (True, reward, ended) + ((lengths,) if return_length else ())

# Define function to return image
def _label_with_episode_number(frame, episode_num, frame_no, p):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    text_color = (255, 255, 255)
    #font = ImageFont.truetype("arial.ttf", size=45)
    drawer.text(
        (100, 5),
        f"Episode: {episode_num+1}     Frame: {frame_no}",
        fill=text_color,
        #font=font,
    )
    if p == 1:
        player = "Player 1"
        color = (255, 0, 0)
    if p == 2:
        player = "Player 2"
        color = (100, 255, 150)
    if p is None:
        player = "Self-play"
        color = (255, 255, 255)
    drawer.text(
        (700, 5), 
        f"Agent: {player}", 
        fill=color, 
        #font=font
    )
    return im


# Resizes frames to make file size smaller
def resize_frames(frames, fraction):
    resized_frames = []
    for img in frames:
        new_width = int(img.width * fraction)
        new_height = int(img.height * fraction)
        img_resized = img.resize((new_width, new_height))
        resized_frames.append(np.array(img_resized))

    return resized_frames


if __name__ == "__main__":
    # device = torch.device('mps') # if on mac
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = "./models/DQN/lesson3_trained_agent.pt"  # Path to saved agent checkpoint

    env = connect_four_v3.env(render_mode="rgb_array")
    env.reset()

    # Configure the algo input arguments
    state_dim = [
        env.observation_space(agent)["observation"].shape for agent in env.agents
    ]
    one_hot = False
    action_dim = [env.action_space(agent).n for agent in env.agents]

    # Pre-process dimensions for pytorch layers
    # We will use self-play, so we only need to worry about the state dim of a single agent
    # We flatten the 6x7x2 observation as input to the agent's neural network
    state_dim = np.zeros(state_dim[0]).flatten().shape
    action_dim = action_dim[0]

    # Instantiate an DQN object
    dqn = DQN(
        state_dim,
        action_dim,
        one_hot,
        device=device,
    )

    # Load the saved algorithm into the DQN object
    dqn.loadCheckpoint(path)

    for opponent_difficulty in ["random", "weak", "strong", "self"]:
        # Create opponent
        if opponent_difficulty == "self":
            opponent = dqn
        else:
            opponent = Opponent(env, opponent_difficulty)

        # Define test loop parameters
        episodes = 2  # Number of episodes to test agent on
        max_steps = (
            500  # Max number of steps to take in the environment in each episode
        )

        rewards = []  # List to collect total episodic reward
        frames = []  # List to collect frames

        print("============================================")
        print(f"Agent: {path}")
        print(f"Opponent: {opponent_difficulty}")

        # Test loop for inference
        for ep in range(episodes):
            if ep / episodes < 0.5:
                opponent_first = False
                p = 1
            else:
                opponent_first = True
                p = 2
            if opponent_difficulty == "self":
                p = None
            env.reset()  # Reset environment at start of episode
            frame = env.render()
            frames.append(
                _label_with_episode_number(frame, episode_num=ep, frame_no=0, p=p)
            )
            observation, reward, done, truncation, _ = env.last()
            player = -1  # Tracker for which player's turn it is
            score = 0
            for idx_step in range(max_steps):
                action_mask = observation["action_mask"]
                if player < 0:
                    state = np.moveaxis(observation["observation"], [-1], [-3])
                    state = np.expand_dims(state, 0)
                    if opponent_first:
                        if opponent_difficulty == "self":
                            action = opponent.getAction(
                                state, epsilon=0, action_mask=action_mask
                            )[0]
                        elif opponent_difficulty == "random":
                            action = opponent.getAction(action_mask)
                        else:
                            action = opponent.getAction(player=0)
                    else:
                        action = dqn.getAction(
                            state, epsilon=0, action_mask=action_mask
                        )[
                            0
                        ]  # Get next action from agent
                if player > 0:
                    state = np.moveaxis(observation["observation"], [-1], [-3])
                    state[[0, 1], :, :] = state[[0, 1], :, :]
                    state = np.expand_dims(state, 0)
                    if not opponent_first:
                        if opponent_difficulty == "self":
                            action = opponent.getAction(
                                state, epsilon=0, action_mask=action_mask
                            )[0]
                        elif opponent_difficulty == "random":
                            action = opponent.getAction(action_mask)
                        else:
                            action = opponent.getAction(player=1)
                    else:
                        action = dqn.getAction(
                            state, epsilon=0, action_mask=action_mask
                        )[
                            0
                        ]  # Get next action from agent
                env.step(action)  # Act in environment
                observation, reward, termination, truncation, _ = env.last()
                # Save the frame for this step and append to frames list
                frame = env.render()
                frames.append(
                    _label_with_episode_number(
                        frame, episode_num=ep, frame_no=idx_step, p=p
                    )
                )

                if (player > 0 and opponent_first) or (
                    player < 0 and not opponent_first
                ):
                    score += reward
                else:
                    score -= reward

                # Stop episode if any agents have terminated
                if truncation or termination:
                    break

                player *= -1

            print("-" * 15, f"Episode: {ep+1}", "-" * 15)
            print(f"Episode length: {idx_step}")
            print(f"Score: {score}")

        print("============================================")

        frames = resize_frames(frames, 0.5)

        # Save the gif to specified path
        gif_path = "./videos/"
        os.makedirs(gif_path, exist_ok=True)
        imageio.mimwrite(
            os.path.join("./videos/", f"connect_four_{opponent_difficulty}_opp.gif"),
            frames,
            duration=400,
            loop=True,
        )

    env.close()