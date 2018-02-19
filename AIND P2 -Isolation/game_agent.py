"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

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
    # Simplest strategy
    # my moves- 2* opp moves
    # Maybe Experiment with variable weight?
    
    if game.is_loser(player):
        return(float("-inf"))
        
    if game.is_winner(player):
        return(float("inf"))
        
    my_moves_count=len(game.get_legal_moves(player))
    opp_moves_count=len(game.get_legal_moves(game.get_opponent(player)))
    score=float(my_moves_count-2*opp_moves_count)
    return score
        

    
    


def custom_score_2(game, player):
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
    
    if game.is_loser(player):
        return(float("-inf"))
        
    if game.is_winner(player):
        return(float("inf"))
        
        
    my_moves_count=len(game.get_legal_moves(player))
    my_moves=game.get_legal_moves(player)
    
    opp_moves_count= float("inf")
    
    for move in my_moves:
        forecasted_game=game.forecast_move(move)
        forecasted_opp_moves_count=forecasted_game.get_legal_moves(game.get_opponent(player))
        
        if not opp_moves_count==float("inf") or len(forecasted_opp_moves_count)<opp_moves_count:
            opp_moves_count=len(forecasted_opp_moves_count)
    
    score= float( my_moves_count-2*opp_moves_count)
    return score
        

    
def custom_score_3(game, player):
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
    if game.is_loser(player):
        return(float("-inf"))
        
    if game.is_winner(player):
        return(float("inf"))
        
        
    
    blank_spaces = len(game.get_blank_spaces())
    my_moves_count = len(game.get_legal_moves(player))
    opp_moves_count=len(game.get_legal_moves(game.get_opponent(player)))
    
    score = (my_moves_count - 2* opp_moves_count)+blank_spaces/2
    return score


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        legal_moves=game.get_legal_moves()
        
        if bool(legal_moves):
            best_move=legal_moves[0]
        else:
            return (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move
    
    ##### Helper functions for minimax algorithm.
    ##### Modeled on the AIMA textbook algorithm
    ##### Reference AIMA Section 5.2.2  Figure 5.3
    
    def min_value(self,game,depth):
        
        ### Timer check as per notes 
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        #Terminal test : Max depth reached? or end of game for player?
        
        if depth==0 or len(game.get_legal_moves())==0:
            return self.score(game,self)
        
        # Highest value in prep for selecting min value to avoid range conflicts
        minimum_value=float("inf")
        
        # Iterate over all moves and select min value from the max values of next level
        for move in game.get_legal_moves():
            minimum_value=min(minimum_value, self.max_value(game.forecast_move(move),depth-1))
        
        return minimum_value
    
    
    def max_value(self,game,depth):
        
        ### Timer check as per notes 
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        #Terminal test : Max depth reached? or end of game for player?
        if depth==0 or len(game.get_legal_moves())==0:
            return self.score(game,self)
        
        # Lowest value in prep for selecting max value to avoid range conflicts
        maximum_value=float("-inf")
        
        # Iterate over all moves and select max value from the min values of next level
        
        for move in game.get_legal_moves():
            maximum_value=max(maximum_value,self.min_value(game.forecast_move(move),depth-1))

        
        return maximum_value
    
    

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            

        # Main minimax algo as per the AIMA text
        
        # Reminder: Root node is a max node
        best_score = float("-inf")
        
        # Initial coordinates for best move
#        get_legal_moves(self) will try to always evaluate the moves of your player. in other words, your opponent will ignore their own moves and try to use yours instead.
        legal_moves=game.get_legal_moves()
        
        if bool(legal_moves):
            best_move=legal_moves[0]
        else:
            return (-1, -1)

        # Implement recursive search to find best move and score
        # Reminder: Root node is a max node
        
        for move in game.get_legal_moves():
            temp_score = self.min_value(game.forecast_move(move), depth-1)
            if temp_score > best_score:
                best_score = temp_score
                best_move = move

        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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

        
        #initialize best move coordinates
        legal_moves= game.get_legal_moves()
        
        if bool(legal_moves):
            best_move=legal_moves[0]
        else:
            return (-1,-1)
        
        # Keep searching for best move until time runs out.
        depth=1
        
        try:
            while True:
                temp_move = self.alphabeta(game, depth)
                
                # If no moves are found, retun best move
                if not temp_move:
                    return best_move
                else:
                    # Update best move upon successfull search
                    best_move=temp_move
                # Increment depth to continue searching
                depth+=1
        # return best move upon timeout exception
        except SearchTimeout:
            return best_move

        return best_move
    
    
    #### Implement helper functions and main alphabeta alog as per AIMA text Section 5.3.1 and Fig 5.7
    
    def min_value(self, game, depth, alpha, beta):
        
        ### Timer check as per notes 
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        ### Terminal test : End of game? or max depth reached
        if depth == 0 or len(game.get_legal_moves()) == 0:
            return self.score(game, self)
        
        ### Large value to avoid range conflict
        minimum_value = float("inf")
        
        ### Iterate over all moves, applying every move to the game and deducing min value from the max vals of next level
        ### Update beta at every iteration
        
        for move in game.get_legal_moves():
            minimum_value = min(minimum_value, self.max_value(game.forecast_move(move), depth - 1, alpha, beta))
            if minimum_value <= alpha:
                return minimum_value
            beta = min(beta, minimum_value)
        return minimum_value
    
    
    def max_value(self, game, depth, alpha, beta):
        
        ### Timer check as per notes 
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        ### Terminal test : End of game? or max depth reached
        if depth == 0 or len(game.get_legal_moves()) == 0:
            return self.score(game, self)
        
        ### Smallest value to avoid range conflict
        maximum_value = float("-inf")
        
        ### Iterate over all moves, applying every move to the game and deducing max value from the min vals of next level
        ### Update alpha at every iteration
        for move in game.get_legal_moves():
            maximum_value = max(maximum_value, self.min_value(game.forecast_move(move), depth-1, alpha, beta))
            if maximum_value >= beta:
                return maximum_value
            alpha = max(alpha, maximum_value)
        return maximum_value
    
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

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

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        ##### Main Alpha Beta algo
        
        ## Initialize best move coordinates
#        get_legal_moves(self) will try to always evaluate the moves of your player. in other words, your opponent will ignore their own moves and try to use yours instead.
        legal_moves=game.get_legal_moves()
        
        if bool(legal_moves):
            best_move=legal_moves[0]
        else:
            return (-1, -1)
        ## Reminder : root node is a max node
        best_score = float("-inf")

        ## Main recursive search to run alpha beta and deduce best score and move
        ## root node is a max node
        
        for move in game.get_legal_moves():
            temp_score = self.min_value(game.forecast_move(move), depth-1, alpha, beta)
            if temp_score > best_score:
                best_score = temp_score
                best_move = move
            if best_score>=beta:
                return best_move
            alpha = max(alpha, temp_score)

        return best_move
