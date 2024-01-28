from utils import *
from prophets import *

def Scenario_1(train_set, test_set):
    """
    Question 1.
    2 Prophets 1 Game.
    You may change the input & output parameters of the function as you wish.
    """
    prophet_a = Prophet(0.2) # "The better Prophet" - lower error rate
    prophet_b = Prophet(0.4)
    prophet_a_sel = 0 # How many times we selected Prophet A (the better prophet)
    accumelated_err = 0 # The error accumelated during the testing phase of all experiments
    estimation_err = 0 # The difference of prediction error in all experiments

    experiment_count = 100 # As specified in the exercise document
    for exper_iter in range(experiment_count):
        train_game = np.random.choice(train_set[exper_iter], size=1)

        # Calculating the emperical error of each Prophet
        prophet_a_err = compute_error(prophet_a.predict(train_game), train_game)
        prophet_b_err = compute_error(prophet_b.predict(train_game), train_game)
        
        selected_prophet = None
        # Apply ERM
        if prophet_a_err <= prophet_b_err:
            selected_prophet = prophet_a
            prophet_a_sel += 1
        else:
            selected_prophet = prophet_b
            # The difference between A and B is 0.2, therefore we accumelate it
            # for all experiments, finally we'd be able to calculate mean value
            estimation_err += 0.2 

        # Adding error of the test phase to the rest of the accumelated error
        accumelated_err += compute_error(
                                selected_prophet.predict(test_set), test_set)
        
    # Experiments are done - Now for the results *drumroll*
    print(f"Average error: {accumelated_err / experiment_count}")
    print(f"Best Prophet selections: {prophet_a_sel} out of {experiment_count} experiments")
    # Note: This is constant due to the expermintation setting
    print("Average approx. error: 0.2") 
    print(f"Averge estimation error: {estimation_err / experiment_count}")

def Scenario_2():
    """
    Question 2.
    2 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    pass


def Scenario_3():
    """
    Question 3.
    500 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    pass


def Scenario_4():
    """
    Question 4.
    500 Prophets 1000 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    pass


def Scenario_5():
    """
    Question 5.
    School of Prophets.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    pass


def Scenario_6():
    """
    Question 6.
    The Bias-Variance Tradeoff.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    pass


if __name__ == '__main__':
    np.random.seed(0)  # DO NOT MOVE / REMOVE THIS CODE LINE!

    train_set = create_data(100, 1000)
    test_set = create_data(1, 1000)[0]

    print(f'Scenario 1 Results:')
    Scenario_1(train_set, test_set)

    print(f'Scenario 2 Results:')
    Scenario_2()

    print(f'Scenario 3 Results:')
    Scenario_3()

    print(f'Scenario 4 Results:')
    Scenario_4()

    print(f'Scenario 5 Results:')
    Scenario_5()

    print(f'Scenario 6 Results:')
    Scenario_6()

