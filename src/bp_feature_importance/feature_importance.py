# %%
import numpy as np
import pandas as pd
import random
import itertools
import math
import concurrent.futures
import sys

from bp_dependency import bin_data, unordered_bp_dependency


__all__ = ['bp_feature_importance']  # !


def _flatten(list_of_lists):
    """
    To flatten a list of lists
    """
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return _flatten(list_of_lists[0]) + _flatten(list_of_lists[1:])
    return list_of_lists[:1] + _flatten(list_of_lists[1:])


def _flatten_and_np_array(list_of_lists):
    """
    Flatten a list of lists and return as numpy array
    """
    return np.asarray(_flatten(list_of_lists))

# %%


class _feature_importance_class:

    def __init__(self, dataset, X_indices, Y_indices, binning_indices = None, binning_strategy = 'auto',sequence_strategy = 'exhaustive', stopping_strategy = None, midway_binning = True) -> None:
        """
        This class is used to determine the Berkelmans-Pries Feature Importance (BP-FI)

        Args:
        --------
             dataset (array_like): Dataset that is used to compute the BP-FI values
             X_indices (array_like): 1-dimensional list /numpy.ndarray containing the indices for the X variable.
             Y_indices (array_like): 1-dimensional list / numpy.ndarray containing the indices for the Y variable.
             binning_indices (array_like, optional): 1-dimensional list / numpy.ndarray containing the indices that need to be binned. Default is `None`, which means that no variables are binned.
             binning_strategy (int or sequence of scalars or str, optional): Default is `auto`. See numpy.histogram_bin_edges for more information
             midway_binning (bool, optional): Default is `True`. Determines if the values are reduced to the midways of the bins (if True) or just the index of the bins (if False)
             sequence_strategy (str, optional): Default is `exhaustive`, which samples all possible sequences. Alternative option is `random`, which samples a random sequence every time
             stopping_strategy (int, optional): Default is `None`. Used to stop earlier after x rounds.

        See also:
        --------
            bp_feature_importance, unordered_bp_dependency, bin_data
        """
        #----Initialization----#
        
        # Input variables
        self.dataset = np.asarray(dataset.copy())
        self.X_indices = X_indices
        self.X_indices = [X_index if isinstance(X_index, list) else [X_index] for X_index in X_indices]
        self.Y_indices = np.asarray(Y_indices)
        self.stopping_strategy = stopping_strategy
        self.sequence_strategy = sequence_strategy
        self.binning_indices = binning_indices
        self.binning_strategy = binning_strategy
        self.midway_binning = midway_binning

        if self.binning_indices is not None:
            self.binning_indices = np.asarray(self.binning_indices)
            if not isinstance(self.binning_strategy, dict):
                strat = self.binning_strategy
                if self.binning_indices.ndim == 0:
                    self.binning_strategy = {binning_indices: strat}
                else:
                    self.binning_strategy = {bin_index: strat for bin_index in self.binning_indices}

            # Binning the data
            for bin_index in binning_indices:
                self.dataset[:, bin_index] = bin_data(x=self.dataset[:, bin_index], bins=self.binning_strategy[bin_index], midways=self.midway_binning)

        # Dependency specific:
        self.UD_Y_Y = unordered_bp_dependency(dataset=self.dataset, X_indices=self.Y_indices,
                                              Y_indices=self.Y_indices, binning_indices=None, format_input=False)
        self.UD_all_X_Y = unordered_bp_dependency(dataset=self.dataset, X_indices=_flatten_and_np_array(
            self.X_indices), Y_indices=self.Y_indices, binning_indices=None, format_input=False)
        self.UD_before = 0
        self.UD_after = 0

        # TODO: Ergens checken of dingen niet constant zijn

        # Standard variables
        self.n_sequences_counter = -1
        self.total_sequence = None
        self.current_sequence = None
        self.current_sequence_variable = None
        self.current_sequence_counter = -1
        self.new_ud_value = None
        self.average_shapley_values = {frozenset(index): 0 for index in self.X_indices}
        self.average_shapley_values_counters = {frozenset(index): 0 for index in self.X_indices}
        # dict that keeps track of which ud dependencies have already been computed
        self.computed_ud_dependencies = {frozenset(_flatten_and_np_array(self.X_indices)): self.UD_all_X_Y}

        # strategy specific
        if self.sequence_strategy == 'exhaustive':
            self.sequence_array = list(itertools.permutations(self.X_indices))
            self.stopping_strategy = math.factorial(len(self.X_indices))


    #----Functions----#
    
    def _stop_generating_sequences(self) -> bool:
        """
        Function to stop generation of sequences
        """
        if isinstance(self.stopping_strategy, int):
            # In this case, we want to generate exactly stopping_strategy sequences (that is why -1)
            return(self.n_sequences_counter >= self.stopping_strategy - 1)

    # Functie die bepaald of er gestopt moet worden met 1 sequence

    def _early_sequence_stopping(self) -> bool:
        """
        Function to stop early in the sequence
        """
        # early sequence stopping not implemented yet
        return(False)

    def _update_shapley_value(self):
        """
        Function to update the Shapley values after a new Shapley value has been determined 
        """
        # running average
        self.average_shapley_values[frozenset(self.current_sequence_variable)] = (self.average_shapley_values_counters[frozenset(self.current_sequence_variable)] * self.average_shapley_values[frozenset(
            self.current_sequence_variable)] + self.new_ud_value) / (self.average_shapley_values_counters[frozenset(self.current_sequence_variable)] + 1)
        # update counter
        self.average_shapley_values_counters[frozenset(self.current_sequence_variable)] += 1

   
    

    def _determine_added_value(self):
        """
        Function to determine the added value 
        """
        try:
            # If ud has been previously computed
            self.UD_after = self.computed_ud_dependencies[frozenset(_flatten_and_np_array(self.current_sequence))]
        except:
            # If ud is new
            
            if self._early_sequence_stopping() == True:
                # early sequence stopping is not yet implemented
                print("This is not yet implemented")
                
            else:
                # Determine new ud
                self.UD_after = unordered_bp_dependency(dataset=self.dataset, X_indices=_flatten_and_np_array(
                    self.current_sequence), Y_indices=self.Y_indices, binning_indices=None, format_input=False)
                # Update computed dependencies dict to save time
                self.computed_ud_dependencies[frozenset(_flatten_and_np_array(self.current_sequence))] = self.UD_after

        # Added value
        self.new_ud_value = self.UD_after - self.UD_before

        # For next variable update UD_before
        self.UD_before = self.UD_after


    def _generate_shapley_sequence(self):
        """
        Function to generate a sequence of variables from the X_indices 
        """
        # Count the number of sequences
        self.n_sequences_counter += 1

        # Strategy: Do every possible sequence once
        if self.sequence_strategy == 'exhaustive':
            return_sequence = list(self.sequence_array[self.n_sequences_counter])
            self.total_sequence = return_sequence
            
        # Strategy: Randomly sample a sequence
        if self.sequence_strategy == 'random':
            index_list = self.X_indices.copy()
            random.shuffle(index_list)
            self.total_sequence = index_list



    def _reset_for_new_sequence(self):
        """
        Function to reset variables for a new sequence
        """
        self.current_sequence_counter = -1
        self.UD_after = 0
        self.UD_before = 0
        self.current_sequence = self.total_sequence[:1]
        self.current_sequence_variable = self.total_sequence[0]

    def _next_variable_sequence(self):
        """
        Function to get the next variable and update the current sequence 
        """
        self.current_sequence_counter += 1
        # This is the running sequence
        self.current_sequence = self.total_sequence[: self.current_sequence_counter + 1]
        self.current_sequence_variable = self.total_sequence[self.current_sequence_counter]

    def _divide_average_shapley_by_Y(self):
        """
        To convert the ud values into bp-dependency values 
        """
        self.average_shapley_values = {frozenset(key): value / self.UD_Y_Y for key, value in self.average_shapley_values.items()}


# %%

def bp_feature_importance(dataset, X_indices, Y_indices, binning_indices = None, binning_strategy = 'auto',sequence_strategy = 'exhaustive', stopping_strategy = None, midway_binning = True, print_stats = False):
    """
    To determine the Berkelmans-Pries Feature Importance (BP-FI)

    Args:
    --------
            dataset (array_like): Dataset that is used to compute the BP-FI values
            X_indices (array_like): 1-dimensional list /numpy.ndarray containing the indices for the X variable.
            Y_indices (array_like): 1-dimensional list / numpy.ndarray containing the indices for the Y variable.
            binning_indices (array_like, optional): 1-dimensional list / numpy.ndarray containing the indices that need to be binned. Default is `None`, which means that no variables are binned.
            binning_strategy (int or sequence of scalars or str, optional): Default is `auto`. See numpy.histogram_bin_edges for more information
            midway_binning (bool, optional): Default is `True`. Determines if the values are reduced to the midways of the bins (if True) or just the index of the bins (if False)
            sequence_strategy (str, optional): Default is `exhaustive`, which samples all possible sequences. Alternative option is `random`, which samples a random sequence every time
            stopping_strategy (int, optional): Default is `None`. Used to stop earlier after x rounds.
            print_stats (bool): Default is `False`. Print statistics
            
    Returns:
    --------
       dict: Berkelmans-Pries Feature Importance for each index in X_indices        
            
    See also:
    --------
        _feature_importance_class, unordered_bp_dependency, bin_data
        
    Example:
    --------
        A dataset where Y is the XOR function of X_0 and X_1:
        >>> dataset, X_indices, Y_indices = (np.array([[0,0,0], [1,0,1], [0,1,1], [1,1,0]]), [0,1], [2])
        >>> print(bp_feature_importance(dataset, X_indices, Y_indices))
        {0: 0.5, 1: 0.5}
    """
    
    #----Initialization----#
    
    bp_class = _feature_importance_class(
        dataset=dataset,
        X_indices=X_indices,
        Y_indices=Y_indices,
        binning_indices=binning_indices,
        binning_strategy=binning_strategy,
        midway_binning=midway_binning,
        sequence_strategy=sequence_strategy,
        stopping_strategy=stopping_strategy
    )


    #----Calculate----#
    
    # Keep generating sequences until stopped
    while(bp_class._stop_generating_sequences() == False):
        bp_class._generate_shapley_sequence()
        bp_class._reset_for_new_sequence()
        
        # Go through each variable of the sequence
        for _ in range(len(bp_class.total_sequence)):
            bp_class._next_variable_sequence()
            # determine the added value
            bp_class._determine_added_value()
            # update running averages
            bp_class._update_shapley_value()



    if print_stats == True:
        ud_values = {list(i)[0]: j for i,j in bp_class.average_shapley_values.items()}
        print('Average UD Shapley {}'.format(ud_values))
        print('Which sums up to: {}'.format(sum(bp_class.average_shapley_values.values())))

    # To go from ud to bp-dep, we must divide by ud(Y,Y)
    bp_class._divide_average_shapley_by_Y()

    # To remove the frozensets
    bp_class.average_shapley_values = {list(i)[0]: j for i,j in bp_class.average_shapley_values.items()}

    if print_stats == True:
        print("Average Dependency Shapley {}".format(bp_class.average_shapley_values))
        print('Which sums up to: {}'.format(sum(bp_class.average_shapley_values.values())))
        print('UD of all X_variables: {}'.format(bp_class.UD_all_X_Y))
        print('UD of Y, Y: {}'.format(bp_class.UD_Y_Y))
        print('Dependency of all X_variables: {}'.format(bp_class.UD_all_X_Y / bp_class.UD_Y_Y))
        print("Number of sequences: {}".format(bp_class.n_sequences_counter + 1))
        # To remove the frozensets
        bp_class.computed_ud_dependencies = {list(i)[0]: j for i,j in bp_class.computed_ud_dependencies.items()}
        print("Computed ud values: {}".format(bp_class.computed_ud_dependencies))
        
    return(bp_class.average_shapley_values)


# %%
dataset, X_indices, Y_indices = (np.array([[0,0,0], [1,0,1], [0,1,1], [1,1,0]]), [0,1], [2])
print(bp_feature_importance(dataset, X_indices, Y_indices))

#%%
