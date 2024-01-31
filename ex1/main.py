from utils import *
from prophets import *

def _find_best_prophet(prophets):
    """
    Finding the best prophet based on the error probability
    """
    return min(prophets, key=lambda p: p._err_prob)

def _base_scenario(train_set, test_set, prophets, game_count, prophet_set_b=None):
    """
    The base scenario for all scenarios.
    
    :param train_set: The training set
    :param test_set: The test set
    :param prophets: The prophets to train on
    :param game_count: The number of games to sample from the training set
    """
    best_prophet = _find_best_prophet(prophets)
    best_prophet_selections = 0 # How many times we selected the better prophet
    accumelated_err = 0 # The error accumelated during the testing phase of all experiments
    estimation_err = 0 # The difference of prediction error in all experiments
    # How many times the selected prophet was out of the estimation threshold
    # The current estimation threshold is defined below and is 0.01 (1%)
    out_estimation_threshold = 0 

    experiment_count = 100 # As specified in the exercise document
    for exper_iter in range(experiment_count):
        train_game = np.random.choice(train_set[exper_iter], size=game_count)

        cur_optimal_err = 2 # Initializing with an invalid probability value
        cur_optimal_prophet = None
        cur_est_error = 0
        # Calculating the emperical error of each Prophet
        for prophet in prophets:
            prophet_err = compute_error(prophet.predict(train_game), train_game)
            # Applying ERM to find the best prophet
            if prophet_err < cur_optimal_err:
                cur_optimal_err = prophet_err
                cur_optimal_prophet = prophet

        # If the best prophet has been indeed selected, then we indicate it,
        # otherwise we handle the estimation error
        if cur_optimal_prophet._err_prob == best_prophet._err_prob:
            best_prophet_selections += 1
        else:
            cur_est_error = (cur_optimal_prophet._err_prob - best_prophet._err_prob)
            estimation_err += cur_est_error

        # Checking if the selected prophet is outside of the estimation threshold
        if 0.01 < cur_est_error:
            out_estimation_threshold += 1

        # Adding error of the test phase to the rest of the accumelated error
        accumelated_err += compute_error(
                                cur_optimal_prophet.predict(test_set), test_set)
    
    # Experiments are done - Now for the results *drumroll*
    print(f"Average error: {accumelated_err / experiment_count}")
    print(f"Best Prophet selections: {best_prophet_selections} out of {experiment_count} experiments")
    print(f"Average approximation error: {best_prophet._err_prob}") 
    print(f"Average estimation error: {estimation_err / experiment_count}")
    print(f"Prophet selections out of 1% estimation threshold: {out_estimation_threshold} out of {experiment_count} experiments") 

def _experiment_on_prophet_set(train_game, prophet_set, best_prophet):
    """
    """
    estimation_err = 0
    emperical_test_error = 0
    cur_optimal_err = 2 # Invalid probability value
    cur_optimal_prophet = None

    # Calculating the emperical error of each Prophet
    for prophet in prophet_set:
        prophet_err = compute_error(prophet.predict(train_game), train_game)
        # Applying ERM to find the best prophet
        if prophet_err < cur_optimal_err:
            cur_optimal_err = prophet_err
            cur_optimal_prophet = prophet

    # If the best prophet has been indeed selected, then we indicate it,
    # otherwise we handle the estimation error
    if cur_optimal_prophet._err_prob != best_prophet._err_prob:
        estimation_err = (cur_optimal_prophet._err_prob - best_prophet._err_prob)

    # Adding error of the test phase to the rest of the accumelated error
    emperical_test_error = compute_error(
                            cur_optimal_prophet.predict(test_set), test_set)
    
    return (emperical_test_error, estimation_err)

def _prophet_classes_scenario(train_set, test_set, set_a, set_b, game_count):
    """
    The base scenario for all scenarios.
    
    :param train_set: The training set
    :param test_set: The test set
    :param set_a: The first set of prophets to train on
    :param set_b: The second set of prophets to train on
    :param game_count: The number of games to sample from the training set
    """
    best_prophet_a = _find_best_prophet(set_a)
    best_prophet_b = _find_best_prophet(set_b)
    accumelated_err_a = 0
    estimation_err_a = 0
    accumelated_err_b = 0
    estimation_err_b = 0

    experiment_count = 100 # As specified in the exercise document
    for exper_iter in range(experiment_count):
        train_game = np.random.choice(train_set[exper_iter], size=game_count)

        exp_err_a, exp_est_err_a = _experiment_on_prophet_set(train_game, set_a, best_prophet_a)
        accumelated_err_a += exp_err_a
        estimation_err_a += exp_est_err_a

        exp_err_b, exp_est_err_b = _experiment_on_prophet_set(train_game, set_b, best_prophet_b)
        accumelated_err_b += exp_err_b
        estimation_err_b += exp_est_err_b
        
    # Experiments are done - Now for the results *drumroll*
    print("Prophet Set A results:")
    print(f"Average error: {accumelated_err_a / experiment_count}")
    print(f"Average approximation error: {best_prophet_a._err_prob}") 
    print(f"Average estimation error: {estimation_err_a / experiment_count}")

    print("Prophet Set B results:")
    print(f"Average error: {accumelated_err_b / experiment_count}")
    print(f"Average approximation error: {best_prophet_b._err_prob}") 
    print(f"Average estimation error: {estimation_err_b / experiment_count}")


def Scenario_1(train_set, test_set):
    """
    Question 1.
    2 Prophets 1 Game.
    You may change the input & output parameters of the function as you wish.
    """
    prophet_a = Prophet(0.2) # "The better Prophet" - lower error rate
    prophet_b = Prophet(0.4)
    _base_scenario(train_set, test_set, [prophet_a, prophet_b], 1)

def Scenario_2(train_set, test_set):
    """
    Question 2.
    2 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    prophet_a = Prophet(0.2) # "The better Prophet" - lower error rate
    prophet_b = Prophet(0.4)
    _base_scenario(train_set, test_set, [prophet_a, prophet_b], 10)

def Scenario_3(train_set, test_set):
    """
    Question 3.
    500 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    
    prophets = sample_prophets(500, 0, 1) # Original scenario [0,1] rates
    #prophets = sample_prophets(500, 0, 0.5) # [0,0.5] Error rates scenario
    _base_scenario(train_set, test_set, prophets, 10)

def Scenario_4(train_set, test_set):
    """
    Question 4.
    500 Prophets 1000 Games.
    You may change the input & output parameters of the function as you wish.
    """
    prophets = sample_prophets(500, 0, 1)
    _base_scenario(train_set, test_set, prophets, 1000)

def Scenario_5(train_set, test_set):
    """
    Question 5.
    School of Prophets.
    You may change the input & output parameters of the function as you wish.
    """
    prophet_counts = [2,5,10,50] # Defined as 'k' in the exercise document
    games_count = [1,10,50,1000] # Defined as 'm' in the exercise document

    for prophet_count in prophet_counts:
        for game_count in games_count:
            print(f"[*] Sub-Scenario - ({prophet_count}, {game_count})")
            prophets = sample_prophets(prophet_count, 0, 0.2)
            _base_scenario(train_set, test_set, prophets, game_count)

def Scenario_6(train_set, test_set):
    """
    Question 6.
    The Bias-Variance Tradeoff.
    You may change the input & output parameters of the function as you wish.
    """
    set_a = sample_prophets(5, 0.3, 0.6)
    set_b = sample_prophets(500, 0.25, 0.6)
    _prophet_classes_scenario(train_set, test_set, set_a, set_b, 10)

if __name__ == '__main__':
    np.random.seed(0)  # DO NOT MOVE / REMOVE THIS CODE LINE!

    train_set = create_data(100, 1000)
    test_set = create_data(1, 1000)[0]

    print(f'Scenario 1 Results:')
    Scenario_1(train_set, test_set)

    print(f'Scenario 2 Results:')
    Scenario_2(train_set, test_set)

    print(f'Scenario 3 Results:')
    Scenario_3(train_set, test_set)

    print(f'Scenario 4 Results:')
    Scenario_4(train_set, test_set)

    print(f'Scenario 5 Results:')
    Scenario_5(train_set, test_set)

    print(f'Scenario 6 Results:')
    Scenario_6(train_set, test_set)

