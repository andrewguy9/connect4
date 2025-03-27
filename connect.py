from typing import Callable, List, Literal, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

#########
# BOARD #
#########

Board = torch.Tensor
Move = int
PlayerId = int # Was Literal[-1, 1], but torch couldn't understand it.

# 2. Check for a win using convolution for efficiency.
@torch.jit.script
def check_win(board: Board, player: PlayerId) -> bool:
    # Create a binary mask: 1.0 where board equals player, 0 otherwise.
    b = (board == player).to(torch.float32)
    # Add batch and channel dimensions: shape (1, 1, 6, 7)
    b = b.unsqueeze(0).unsqueeze(0)

    # Define convolution kernels for horizontal, vertical and two diagonal directions.
    kernel_h = torch.tensor([[[[1.0, 1.0, 1.0, 1.0]]]])
    kernel_v = torch.tensor([[[[1.0], [1.0], [1.0], [1.0]]]])
    kernel_d1 = torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel_d2 = torch.flip(kernel_d1, dims=[-1])

    # Apply convolutions; the output dimensions will be reduced accordingly.
    conv_h = F.conv2d(b, kernel_h, stride=1)
    conv_v = F.conv2d(b, kernel_v, stride=1)
    conv_d1 = F.conv2d(b, kernel_d1, stride=1)
    conv_d2 = F.conv2d(b, kernel_d2, stride=1)

    # Check if any convolution output equals 4, which would indicate four in a row.
    if conv_h.eq(4).any().item():
        return True
    if conv_v.eq(4).any().item():
        return True
    if conv_d1.eq(4).any().item():
        return True
    if conv_d2.eq(4).any().item():
        return True

    return False

# 3. Valid move mask: returns a (7,) bool tensor indicating playable columns.
@torch.jit.script
def valid_move_mask(board: torch.Tensor) -> torch.Tensor:
    # A move is valid if the top row in the column is 0 (empty).
    return board[0, :] == 0

# 4. Player move: drop a piece into the selected column.
@torch.jit.script
def make_move(board: Board, col: Move, player: PlayerId) -> Board:
    # First, check if the column is full.
    if board[0, col] != 0:
        raise ValueError("Column is full")

    new_board = board.clone()
    rows = board.size(0)
    # Loop from the bottom row upward.
    for i in range(rows):
        # Calculate row index from the bottom: rows - 1 - i
        row = rows - 1 - i
        if new_board[row, col] == 0:
            new_board[row, col] = player
            return new_board
    # Should never reach here if the column is not full.
    raise RuntimeError("Failed to make move")

# 5. Check for stalemate (i.e. board full at top row for every column)
@torch.jit.script
def is_stalemate(board: torch.Tensor) -> bool:
    return bool(board[0, :].ne(0).all())

@torch.jit.script
def best_move(policy: torch.Tensor, board: Board) -> Move:
    """
    Take a policy vector and a board, and return the best possible move (a column index)
    """
    valid = valid_move_mask(board)
    # Sum valid mask to check if there are any valid moves
    if valid.sum() == 0:
        raise ValueError("No valid moves available")
    
    # Copy policy and set invalid moves to -inf so they are not chosen.
    masked_policy = policy.clone()
    for i in range(masked_policy.size(0)):
        if not valid[i]:
            masked_policy[i] = float('-inf')
    
    # Pick the index with the highest score.
    col = int(torch.argmax(masked_policy))
    return col

@torch.jit.script
def column_height(board: torch.Tensor, col: int) -> int:
    # Returns the row index where a piece would land in column 'col'
    rows = board.size(0)
    for i in range(rows):
        r = rows - 1 - i
        if board[r, col] == 0:
            return r
    return -1  # Should not happen if column is valid

@torch.jit.script
def connectivity_score(board: torch.Tensor, row: int, col: int, player: int) -> int:
    # Check four directions: horizontal, vertical, and the two diagonals.
    max_count = 1  # Count the newly placed piece
    # Define directions: (drow, dcol)
    directions = torch.tensor([[0, 1], [1, 0], [1, 1], [1, -1]])
    for d in range(4):
        drow = int(directions[d, 0].item())
        dcol = int(directions[d, 1].item())
        count = 1
        
        # Look in the positive direction.
        for step in range(1, 7):  # board is 6 rows max.
            r = row + step * drow
            c = col + step * dcol
            if r < 0 or r >= board.size(0) or c < 0 or c >= board.size(1) or board[r, c] != player:
                break
            count += 1
        
        # Look in the negative direction.
        for step in range(1, 7):
            r = row - step * drow
            c = col - step * dcol
            if r < 0 or r >= board.size(0) or c < 0 or c >= board.size(1) or board[r, c] != player:
                break
            count += 1
        
        if count > max_count:
            max_count = count
    return max_count

# Updated best move: choose the highest scoring valid column from a policy vector.
@torch.jit.script
def make_best_move(board: torch.Tensor, policy: torch.Tensor, player: int = 1) -> torch.Tensor:
    return make_move(board, best_move(policy, board), player)

def pretty_print_board(board: torch.Tensor) -> None:
    # Convert board to list for easy iteration.
    board_list = board.tolist()
    # Define symbols for each value.
    symbol_map = {0: '.', 1: 'X', -1: 'O'}
    
    # Print each row.
    for row in board_list:
        print(" ".join(symbol_map[cell] for cell in row))

###########
# PLAYERS #
###########

Context = torch.Tensor
PlayerName = Literal["random", "rank", "greedy", "net"]
MoveFn = Callable[[Board, Context, PlayerId], Move]
ResetFn = Callable[[], Context]
PlayerTuple = Tuple[MoveFn, ResetFn, PlayerName]
PlayerFn = Callable[[], PlayerTuple]

def make_random_player() -> PlayerTuple:
    @torch.jit.script
    def reset_random_player():
        return torch.zeros(0)
    
    @torch.jit.script
    def random_player_move(board: Board, ctx: Context, player: int = 1) -> Move:
        # Create a random policy vector of size (7,) with values in [0, 1).
        policy = torch.rand((7,))
        col = best_move(policy, board)
        return col

    return random_player_move, reset_random_player, "random"

def make_random_ranked_player() -> PlayerTuple:
    @torch.jit.script
    def reset_rand_ranked_player() -> Context:
        # Generate a random permutation of columns (0 to 6)
        random_order = torch.randperm(7)
        return random_order

    @torch.jit.script
    def randomized_ranked_player_move(board: Board, policy: Context, player: PlayerId) -> Move:
        return best_move(policy, board)

    return randomized_ranked_player_move, reset_rand_ranked_player, "rank"

def make_greedy_player():
    @torch.jit.script
    def reset_greedy_player() -> torch.Tensor:
        # This player does not need per-game context, so we return an empty tensor.
        return torch.zeros(0)
    
    @torch.jit.script
    def greedy_player_move(board: torch.Tensor, ctx: torch.Tensor, player: int = 1) -> Move:
        # 1. Check for an immediate winning move.
        for col in range(7):
            if board[0, col] == 0:
                board_after = make_move(board, col, player)
                if check_win(board_after, player):
                    return col

        # 2. Block the opponent's immediate winning move.
        opponent = -player
        for col in range(7):
            if board[0, col] == 0:
                board_after = make_move(board, col, opponent)
                if check_win(board_after, opponent):
                    return col

        # Phase 3: Evaluate moves based on connectivity using tensors.
        # Initialize connectivity scores with -infinity for invalid moves.
        connectivity_scores = torch.full((7,), float('-inf'))
        valid_mask = valid_move_mask(board)   
        # Compute connectivity scores for each valid column.
        for col in range(7):
            if valid_mask[col]:
                simulated_board = make_move(board, col, player)
                # Since the move is valid, column_height(board, col) should always return a valid row.
                score = connectivity_score(simulated_board, column_height(board, col), col, player)
                connectivity_scores[col] = score

        # Probabilistic move selection using softmax and multinomial sampling.
        probs = torch.softmax(connectivity_scores, dim=0)
        selected_move = int(torch.multinomial(probs, 1).item())
        return selected_move
    
    return greedy_player_move, reset_greedy_player, "greedy"


class Connect4Net(nn.Module):
    def __init__(self):
        super(Connect4Net, self).__init__()
        # Input shape: (batch, 2, 6, 7)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Fully connected layer: we flatten the convolutional features.
        self.fc1 = nn.Linear(64 * 6 * 7, 128)
        # Policy head: outputs a vector for the 7 columns.
        self.fc_policy = nn.Linear(128, 7)
        # Optionally, you could add a value head here if desired.
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch, 2, 6, 7) representing the board state.
        Returns:
            Tensor of shape (batch, 7) with raw policy logits.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Flatten: (batch, 64 * 6 * 7)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        policy_logits = self.fc_policy(x)
        return policy_logits

    @torch.jit.export
    def board_to_nnet_input(self, board: Board, current_player: PlayerId) -> torch.Tensor:
        # Create a binary mask for current player's pieces.
        current_channel = (board == current_player).to(torch.float32)
        # Create a binary mask for opponent's pieces.
        opponent_channel = (board == -current_player).to(torch.float32)
        # Stack the two channels to create a (2, 6, 7) tensor.
        return torch.stack([current_channel, opponent_channel], dim=0)

    @torch.jit.export
    def net_infer_logits(self, board: Board, player: PlayerId)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the action as a tensor, the selected action probability as a tensor, the log probability distrubtion (as a tensor) and the logits masked by valid moves.
        Useful for supervised learning.
        """
        state_input = self.board_to_nnet_input(board, player).unsqueeze(0)  # shape (1,2,6,7)
        logits = self(state_input)[0]  # shape (7,)
        
        # Mask invalid moves by setting them to -infinity so they have zero probability.
        valid = valid_move_mask(board)
        masked_logits = logits.clone()
        masked_logits[~valid] = float('-inf')
        probs = F.softmax(masked_logits, dim=0)
        action = torch.multinomial(probs, num_samples=1)
        log_probs = F.log_softmax(masked_logits, dim=0)
        selected_log_prob = log_probs[action]

        return action, selected_log_prob, log_probs, masked_logits

    @torch.jit.export
    def net_infer_log_prob(self, board: Board, player: PlayerId)->Tuple[torch.Tensor, torch.Tensor]:
        """
        returns the move as a tensor, and log_prob for the selected move.
        Useful for reinforcement learning.
        """
        action_tensor, selected_prob, log_prob, masked_logits = self.net_infer_logits(board, player)
        return action_tensor, selected_prob

    @torch.jit.export
    def net_infer_move(self, board: Board, player: PlayerId) -> Move:
        """
        Evaluation function for the net. Returns integer move decision.
        """
        action_tensor, _ = self.net_infer_log_prob(board, player)
        return int(action_tensor)

# TODO use?
scripted_net = torch.jit.script(Connect4Net())

# TODO i bet this will not be passable into script, because it relies on a shared scope.
def make_net_player(net: Connect4Net) -> PlayerTuple:

    def net_reset() -> Connect4Net:
        return net
    
    def net_infer(board: Board, net: Connect4Net, player: PlayerId) -> Move:
        return net.net_infer_move(board, player)

    return net_infer, net_reset, "net"

##########
# Loops # 
##########

def faceoff_loop(p1: PlayerTuple, p2: PlayerTuple) -> None:
    # Initialize an empty board (6 rows x 7 columns) with dtype int8.
    board = torch.zeros((6, 7), dtype=torch.int8)
    # Create the neural network (Player 1). It is untrained.
    net = Connect4Net()
    
    p1_move, p1_reset, p1_name = p1
    p1_ctx = p1_reset()
    p2_move, p2_reset, p2_name = p2
    p2_ctx = p2_reset()

    current_player: PlayerId = 1
    winner = 0
    move_num = 0
    
    pretty_print_board(board)
    print("-" * (7*2))
    
    while move_num < 42:
        if is_stalemate(board):
            break
        
        if current_player == 1:
            move = p1_move(board, p1_ctx, current_player)
            # TODO this could be a net or a player, but its working! XXX
            board = make_move(board, move, current_player)
        else:
            move = p2_move(board, p2_ctx, current_player)
            board = make_move(board, move, current_player)
        
        pretty_print_board(board)
        print("-" * 20)
        
        if check_win(board, current_player):
            winner = current_player
            break
        
        current_player = -current_player
        move_num += 1
    
    if winner == 1:
        print(f"Player 1 {p1_name} wins!")
    elif winner == -1:
        print(f"Player 2 {p2_name} wins!")
    else:
        print("Stalemate!")

def train_self_play(net: Connect4Net, optimizer: torch.optim.Optimizer, num_games: int = 1000):
    """
    Train the network using self-play.
    For each game, we record:
      - The state (converted via board_to_nnet_input)
      - The log probability of the move chosen (sampled from the network's policy)
      - The player that made the move
    At the end of the game, we assign a reward of:
      +1 if that move was made by the winner,
      -1 if by the loser,
       0 if the game was a stalemate.
    Then we update the network using the REINFORCE policy gradient loss.
    """
    net.train()
    p1_wins = 0
    p2_wins = 0
    stalemates = 0
    with tqdm(total=num_games, desc="Self play", leave=False) as pbar:
        for game in range(num_games):
            board = torch.zeros((6, 7), dtype=torch.int8)
            current_player = 1  # We'll have the network play both sides.
            
            # These lists will store the trajectory of the game.
            log_probs: List[torch.Tensor] = []
            players: List[int] = []
            
            done = False
            while not done:
                action, log_prob = net.net_infer_log_prob(board, current_player)
                
                # Record the log probability and the current player.
                log_probs.append(log_prob)
                players.append(current_player)
                
                # Make the move.
                board = make_move(board, action, current_player)
                
                # Check for terminal conditions.
                if check_win(board, current_player):
                    # Current player won.
                    final_reward = 1.0
                    done = True
                    if current_player == 1:
                        p1_wins += 1
                    else:
                        p2_wins += 1
                elif is_stalemate(board):
                    final_reward =  0.0
                    done = True
                    stalemates += 1
                else:
                    # No outcome yet; switch player.
                    current_player = -current_player
            
            # Once the game is finished, assign rewards to each move.
            # For every move, if the move was made by the winner, reward=+1;
            # if by the loser, reward=-1; if stalemate, reward=-0.1.
            # Note: When the game is finished with a win, current_player still holds the value of the winning player.
            returns: List[float] = []
            for p in players:
                if final_reward == 0.0:
                    returns.append(-0.1) # If stalemate, reward is -0.1.
                else:
                    # From each move's perspective, reward is +1 if that player equals the winner,
                    # otherwise -1.
                    returns.append(1.0 if p == current_player else -1.0)
            
            # Compute the policy loss.
            scale = 10000 # TODO maybe remove
            loss: torch.Tensor = torch.tensor([0.0])
            base_discount = 0.98
            discount = 1.0
            assert(len(log_probs) == len(returns))
            # TODO dont need to reverse if use the trick.
            for log_prob, r in zip(reversed(log_probs), reversed(returns)):
                loss -= log_prob * scale * r * discount  # maximize log_prob for winning moves
                discount *= base_discount
            # print("loss", loss, "winner", current_player, "had winner", final_reward)
            
            # Update the network.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (game + 1) % 100 == 0:
                pbar.update(100)
                pbar.set_postfix({
                    "loss": loss.item(),
                    "scores:": f"{p1_wins}/{p2_wins}/{stalemates}"
                })


def train_vs_player(net: Connect4Net, optimizer: torch.optim.Optimizer, player: PlayerTuple, num_games: int = 1000):
    """
    Here we let the network play as Player 1 against a random opponent (Player 2).
    Only the moves from the neural network player are used for the policy update.
    """
    p2_move, p2_reset, p2_name = player
    net.train()
    net_wins = 0
    with tqdm(total=num_games, desc=f"net vs {p2_name}", leave=False) as pbar:
        for game in range(num_games):
            board: Board = torch.zeros((6, 7), dtype=torch.int8)
            current_player: PlayerId = 1  # Neural network is always Player 1.
            
            log_probs: List[torch.Tensor] = []
            done = False

            p2_ctx = p2_reset()
            while not done:
                if current_player == 1:
                    action, log_prob = net.net_infer_log_prob(board, current_player)
                    log_probs.append(log_prob)
                    board = make_move(board, action, current_player)
                else:
                    # Algoritm player move
                    move = p2_move(board, p2_ctx, current_player)
                    board = make_move(board, move, current_player)
                
                # Check terminal conditions.
                if check_win(board, current_player):
                    winner = current_player
                    done = True
                elif is_stalemate(board):
                    winner = 0
                    done = True
                else:
                    # Switch only if game is not over.
                    current_player = -current_player
            
            # Determine reward for neural network's moves.
            # For a win, reward +1; for a loss, -1; draw gives 0.
            if winner == 1:
                reward = 1.0
                net_wins += 1
            elif winner == -1:
                reward = -1.0
            else:
                reward = -0.1
            
            scale = 200
            loss: torch.Tensor = torch.Tensor([0.0])
            discount = 1.0
            discount_rate = 0.9
            for log_prob in reversed(log_probs):
                loss -= log_prob * scale * reward * discount
                discount *= discount_rate
            
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            
            if (game + 1) % 100 == 0:
                pbar.update(100)
                pbar.set_postfix({
                    "loss": loss.item(),
                    "Win%": f"{net_wins/(game+1):3.0%}",
                })

def train_supervised_vs_functions(net: Connect4Net, 
                                  optimizer: torch.optim.Optimizer, 
                                  trainer: PlayerTuple,
                                  opponent: PlayerTuple,
                                  num_games: int,
                                  discount_rate: float = 0.9):
    """
    Trains the network using supervised learning by imitating a trainer's moves (Player 1) 
    while playing against an opponent (Player 2).

    Parameters:
      net: The neural network model.
      optimizer: The optimizer used for training.
      trainer: Tuple (move_fn, reset_fn, name) for the trainer.
      opponent: Tuple (move_fn, reset_fn, name) for the opponent.
      num_games: Number of games to simulate.
      discount_rate: A discount factor (0 < discount_rate <= 1) that weights earlier moves less than later moves.
                     The final move gets a weight of 1, and earlier moves get weight discount_rate**(T-1-i)
                     where T is the number of moves and i is the move index.
                     
    During a game, when it's Player 1's turn, the network's predicted logits (with invalid moves masked)
    are compared to the trainer's move using cross entropy loss.
    
    The loss for each move is stored, and after the game the losses are combined with a discount rate,
    then the network is updated only once after the epoch (all games) is evaluated.
    """
    net.train()
    total_loss = 0.0
    correct_moves = 0
    total_moves = 0

    t_move, t_reset, t_name = trainer
    o_move, o_reset, o_name = opponent

    epoch_loss = torch.tensor(0.0, requires_grad=True)


    with tqdm(total=num_games, desc=f"Supervised Training: Trainer:{t_name} vs Opponent:{o_name}", leave=False) as pbar:
        for game in range(num_games):
            # Initialize an empty board (6 rows x 7 columns) with the network as Player 1.
            board = torch.zeros((6, 7), dtype=torch.int8)
            current_player:PlayerId = 1

            t_ctx = t_reset()
            o_ctx = o_reset()

            move_losses: List[torch.Tensor] = []
            while True:
                if current_player == 1:
                    # Get the trainer's move.
                    trainer_action = t_move(board, t_ctx, current_player)
                    # TODO XXX
                    net_action_tensor, selected_prob, dist, masked_logits = net.net_infer_logits(board, current_player)
                    net_action = int(net_action_tensor)

                    if net_action == trainer_action:
                        correct_moves += 1
                    total_moves += 1

                    # Compute the cross entropy loss using the trainer's action as target.
                    loss = F.cross_entropy(masked_logits.unsqueeze(0), torch.tensor([trainer_action]))

                    move_losses.append(loss)
                    
                    # Update the board using the trainer's move.
                    board = make_move(board, trainer_action, current_player)
                else:
                    opponent_action = o_move(board, o_ctx, current_player)
                    board = make_move(board, opponent_action, current_player)
                
                # Check for terminal conditions.
                if check_win(board, current_player) or is_stalemate(board):
                    break
                else:
                    # Switch the current player.
                    current_player = -current_player

            # Discount the losses. We want the final moves to count more, so
            # we use a factor discount_rate**(T-1-i) for move index i (with T total moves).
            T = len(move_losses)
            if T > 0:
                game_loss = torch.tensor(0.0)
                for i, loss in enumerate(move_losses):
                    weight = discount_rate ** (T - 1 - i)
                    game_loss = game_loss + weight * loss
                epoch_loss = epoch_loss + game_loss
                total_loss += game_loss.item()
            else:
                game_loss = torch.tensor(0.0)

            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{game_loss.item():.4f}" if T > 0 else "N/A",
                "accuracy": f"{(correct_moves/total_moves):.2%}" if total_moves > 0 else "N/A"
            })

    # Update the network once after all games have been evaluated.
    optimizer.zero_grad()
    epoch_loss.backward()
    optimizer.step()

    avg_loss_str = f"{total_loss/num_games:.4f}"
    mv_acc_str = f"{(correct_moves/max(total_moves, 1)):.2%}"
    pbar.write(f"Average loss per game: {avg_loss_str} Overall move accuracy: {mv_acc_str}")

def evaluate_player_vs_player(player1: PlayerTuple, player2: PlayerTuple, num_games: int = 1000) -> Tuple[int, int, int]:
    """
    Evaluate the neural network (as Player 1) against a random player (as Player 2)
    over num_games. Returns a tuple of (net wins, random wins, stalemates).
    """
    p1_mov, p1_reset, p1_name = player1
    p2_mov, p2_reset, p2_name = player2
    p1_wins = 0
    p2_wins = 0
    stalemates = 0
    with torch.no_grad():
        for _ in range(num_games):
            p1_ctx = p1_reset()
            p2_ctx = p2_reset()
            board = torch.zeros((6, 7), dtype=torch.int8)
            current_player: PlayerId = 1
            while True:
                if is_stalemate(board):
                    stalemates += 1
                    break
                if current_player == 1:
                    move: Move = p1_mov(board, p1_ctx, current_player)
                    board = make_move(board, move, current_player)
                else:
                    move = p2_mov(board, p2_ctx, current_player)
                    board = make_move(board, move, current_player)
                    # board = random_player_move(board, player=current_player)
                if check_win(board, current_player):
                    if current_player == 1:
                        p1_wins += 1
                    else:
                        p2_wins += 1
                    break
                current_player = -current_player
    return p1_wins, p2_wins, stalemates


def evaluation_main() -> None:
    net = Connect4Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    n_eval = 1000
    epochs_supervised = 200
    epochs_reinforcement=1000
    n_train_rnd = 100
    n_train_self = 100

    net_player = make_net_player(net)

    faceoffs = True
    supervised = True
    reinforcement = True
    selfplay = True

    r_player = make_random_player()
    rr_player = make_random_ranked_player()
    greed_player = make_greedy_player()

    # Faceoffs
    if faceoffs:
        print("Demo: Random Player")
        faceoff_loop(r_player, r_player)

        print("Demo: Random Ranked Player")
        faceoff_loop(rr_player, r_player)

        print("Demo: Greedy Player")
        faceoff_loop(greed_player, r_player)

        print("-" * 80)
        net.eval()
        print("Demo: Neural Network")
        faceoff_loop(net_player, r_player)

    # SUPERVISED TRAINING
    if supervised:
        for epoch in (pbar:=tqdm(range(epochs_supervised), desc="Supervised")):
            # TODO in train function, we can take a list of opponents or move evaluation out to here like reinforcement.
            train_supervised_vs_functions(net, optimizer, greed_player, r_player, num_games=n_train_rnd)
            #train_supervised_vs_functions(net, optimizer, greed_player, rr_player, num_games=n_train_rnd)
            #train_supervised_vs_functions(net, optimizer, greed_player, greed_player, num_games=n_train_rnd)
            # TODO we are not saving intermediate outputs.

    # REINFORCEMENT TRAINING
    if reinforcement:
        for epoch in (pbar:=tqdm(range(epochs_reinforcement), desc="Reinforcement")):
            train_vs_player(net, optimizer, r_player, num_games=n_train_rnd)
            train_vs_player(net, optimizer, rr_player, num_games=n_train_rnd)
            train_vs_player(net, optimizer, greed_player, num_games=n_train_rnd)

            if epoch % 10 == 0:
                pbar.write(f"--- Epoch {epoch} ---")
                pbar.write("EVALUATING NET VS ALGO PLAYERS:")
                p1_wins, p2_wins, draws = evaluate_player_vs_player(net_player, r_player, num_games=n_eval)
                pbar.write(f"random: {p1_wins}/{p2_wins}/{draws} = {100*p1_wins/n_eval:3.0f}%")

                p1_wins, p2_wins, draws = evaluate_player_vs_player(net_player, rr_player, num_games=n_eval)
                pbar.write(f"ranked: {p1_wins}/{p2_wins}/{draws} = {100*p1_wins/n_eval:3.0f}%")
                
                p1_wins, p2_wins, draws = evaluate_player_vs_player(net_player, greed_player, num_games=n_eval)
                pbar.write(f"greedy: {p1_wins}/{p2_wins}/{draws} = {100*p1_wins/n_eval:3.0f}%")

                torch.save(net.state_dict(), f"connect4_net_{epoch}.pth")
            # pbar.set_postfix({})
    if selfplay:
        for epoch in (pbar:=tqdm(range(epochs_reinforcement), desc="Selfplay")):
            train_self_play(net, optimizer, num_games=n_train_self)

    torch.save(net.state_dict(), f"connect4_net_final.pth")

    # Faceoffs
    if faceoffs:
        print("-" * 80)
        net_player = make_net_player(net)
        print("Demo: Neural Network after training vs random player")
        faceoff_loop(net_player, r_player)

        net_player = make_net_player(net)
        print("Demo: Neural Network after training vs random ranked player")
        faceoff_loop(net_player, rr_player)

        net_player = make_net_player(net)
        print("Demo: Neural Network after training vs greedy player")
        faceoff_loop(net_player, greed_player)

        net_player = make_net_player(net)
        print("Demo: Neural Network after training self-play")
        faceoff_loop(net_player, net_player)

        untrained_net = Connect4Net()
        untrained_net_player = make_net_player(untrained_net)
        print("Demo: Neural Network vs Untrained Neural Network")
        faceoff_loop(net_player, untrained_net_player)

if __name__ == "__main__":
    evaluation_main()
