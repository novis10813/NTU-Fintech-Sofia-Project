class DQNparams:
    '''
    LR : Learning rate of the neural network, not the Q learning's learning rate
    EPSILON : Probability of random walk at first
    MIN_EPSILON : Minimum probability of random walk in the end
    EPISODE : How many times that model will run through the data
    BATCH_SIZE : Batch size of training the data
    MODEL_UPDATE : Update target model every n epoch/iteration
    GAMMA : Discount rate for future Q
    '''
    LR = 0.0001
    EPSILON = 0.1
    MIN_EPSILON = 0.0001
    MEMORY_SIZE = 5000
    EPISODES = 20000
    BATCH_SIZE = 64
    MODEL_UPDATE = 1000
    GAMMA = 0.3