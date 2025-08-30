# create a function diff here

def diff(x, t):
    if len(x) != len(t):
        print("The two lengths are not equal, why would you do that. I am return        ing -1 now")
	return -1
    v = [0]*len(t)
    v[0] = 0
    
    for i in range(1, len(x) - 1):
        v[i] = (x[i] - x[i - 1]) / (t[i] - t[i - 1])
    
    return v
        
