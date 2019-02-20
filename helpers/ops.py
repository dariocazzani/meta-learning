import random 
seed = 42
random.seed(seed)

def shuffle_unison(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a, b