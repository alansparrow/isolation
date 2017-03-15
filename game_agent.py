"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    

    # TODO: finish this function!

    # Play offensively (maximize my future's available moves)
    # (Improved_ID: 87.14%, My_Agent: 85.00%)
    # Heuristic 1: Maximum number of available moves
    # score = len(game.get_legal_moves(player)) * 1.0
    # print("Score: ", score)
    # return score

    # Play defensively (minimize oponent's future's available moves)
    # (Improved_ID: 90%, My_Agent: 67.86%)
    # Heuristic 2: Minimum number of oponent's available moves
    # score = len(game.get_legal_moves(game.get_opponent(player))) * -1.0
    # # print("Score: ", score)
    # return score

    # Play wisely, choose move that maximize my future and minimize my oponent's future 
    # (Improved_ID: 88.57%, My_Agent: 91.43%)
    # Heuristic 3: Maximum of division of number of my moves over oponent's available moves
    score = ((0.1 + (len(game.get_legal_moves(player))) * 1.0) / 
            ((0.1 + len(game.get_legal_moves(game.get_opponent(player)))) * 1.0))
    # print("Score: ", score)
    return score


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        max_score = float("-Inf")
        choosen_move = (-1, -1)
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            # max_score = float("-Inf")
            # choosen_move = (-1, -1)
            # for move in legal_moves:
            #     forecast_game = game.forecast_move(move)
            #     s, m = self.minimax(forecast_game, self.search_depth)
            #     print(legal_moves)
            #     print("\n")
            #     print(x, m)
            #     if (s > max_score):
            #         max_score = s
            #         choosen_move = m
            # return choosen_move
            if (self.iterative):
                iterative_depth = 1
                while (1):
                    s, m = self.minimax(game, iterative_depth, maximizing_player=True)
                    if (s > max_score):
                        max_score = s
                        choosen_move = m
                    iterative_depth += 1
            else:
                s, m = self.minimax(game, self.search_depth, maximizing_player=True)
                if (s > max_score):
                    max_score = s
                    choosen_move = m

        except Timeout:
            # Handle any actions required at timeout, if necessary
            # print(self.time_left())
            pass

        # Return the best move from the last completed search iteration
        return choosen_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        
        max_score = float("-inf")
        min_score = float("inf")
        choosen_move = (-1, -1)
        if (maximizing_player):  # Max node
            legal_moves = game.get_legal_moves()
            for move in legal_moves:
                forecast_game = game.forecast_move(move)
                forecast_score = max_score
                if (depth > 1):
                    forecast_score, forecast_move = self.minimax(forecast_game, depth - 1, not maximizing_player)
                elif (depth == 1):
                    forecast_score = self.score(forecast_game, self)
                if (forecast_score > max_score):
                    max_score = forecast_score
                    choosen_move = move
            return max_score, choosen_move
        else:  # Min node
            legal_moves = game.get_legal_moves()
            for move in legal_moves:
                forecast_game = game.forecast_move(move)
                forecast_score = min_score
                if (depth > 1):
                    forecast_score, forecast_move = self.minimax(forecast_game, depth - 1, not maximizing_player)
                elif (depth == 1):
                    forecast_score = self.score(forecast_game, self)
                if (forecast_score < min_score):
                    min_score = forecast_score
                    choosen_move = move
            return min_score, choosen_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        choosen_move = (-1, -1)
        if (maximizing_player):  # Max node
            legal_moves = game.get_legal_moves()
            forecast_alpha = alpha
            for move in legal_moves:
                forecast_game = game.forecast_move(move)
                if (depth > 1):
                    forecast_alpha, forecast_move = self.alphabeta(forecast_game, depth - 1, alpha, beta, maximizing_player=False)
                elif (depth == 1):
                    forecast_alpha = self.score(forecast_game, self)
                if (forecast_alpha > alpha):  # update better choice
                    alpha = forecast_alpha
                    choosen_move = move
                    if (alpha >= beta):  # prune
                        break
                
            return forecast_alpha, choosen_move
        else:  # Min node
            legal_moves = game.get_legal_moves()
            forecast_beta = beta
            for move in legal_moves:
                forecast_game = game.forecast_move(move)
                if (depth > 1):
                    forecast_beta, forecast_move = self.alphabeta(forecast_game, depth - 1, alpha, beta, maximizing_player=True)
                elif (depth == 1):
                    forecast_beta = self.score(forecast_game, self)
                if (forecast_beta < beta):  # update better choice
                    beta = forecast_beta
                    choosen_move = move
                    if (alpha >= beta):  # prune
                        break
            return forecast_beta, choosen_move
