# create a function diff here

def diff(x, t):
    if len(x) != len(t):
        print("The two lengths are not equal, why would you do that. I am returning -1 now")
        return -1

    v = [0] * len(t)
    v[0] = 0  # No previous value for the first element

    for i in range(1, len(x)):
        v[i] = (x[i] - x[i - 1]) / (t[i] - t[i - 1])

    return v
        
