import numpy as np

def Ufun(x, a, k, m):
    x = np.array(x)
    term1 = k * (np.maximum(x - a, 0) ** m)
    term2 = k * (np.maximum(-x - a, 0) ** m)
    return np.sum(term1 + term2)

def F1(x):
    return np.sum(np.square(x))

def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F3(x):
    dim = len(x)
    return np.sum([np.abs(x[i]) ** (i + 2) for i in range(dim)])

def F4(x):
    dim = len(x)
    return np.sum([np.sum(x[:i+1])**2 for i in range(dim)])

def F5(x):
    return np.max(np.abs(x))

def F6(x):
    dim = len(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def F7(x):
    return np.sum(np.abs(x + 0.5)**2)

def F8(x):
    dim = len(x)
    return np.sum(np.arange(1, dim+1) * (x**4)) + np.random.rand()

def F9(x):
    dim = len(x)
    sum_sq = np.sum(x**2)
    term = np.sum(0.5 * np.arange(1, dim+1) * x)
    return sum_sq + term**2 + term**4

def F10(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def F11(x):
    dim = len(x)
    return 1 + np.sum(np.sin(x)**2) - np.exp(-np.sum(x**2))

def F12(x):
    dim = len(x)
    return 0.5 * np.sum(x**4 - 16*x**2 + 5*x)

def F13(x):
    dim = len(x)
    return np.sum(x**2 - 10*np.cos(2*np.pi*x)) + 10*dim

def F14(x):
    dim = len(x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2)/dim))
    term2 = -np.exp(np.sum(np.cos(2*np.pi*x))/dim)
    return term1 + term2 + 20 + np.exp(1)

def F15(x):
    dim = len(x)
    return (np.sum(x**2)/4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, dim+1)))) + 1

def F16(x):
    dim = len(x)
    term1 = np.sum(np.sin(x)**2)
    term2 = np.exp(-np.sum(x**2)) * np.exp(-np.sum(np.sin(np.sqrt(np.abs(x)))**2))
    return (term1 - term2) * np.exp(-np.sum(np.sin(np.sqrt(np.abs(x)))**2))

def F17(x):
    dim = len(x)
    term1 = 10 * (np.sin(np.pi * (1 + (x[0] + 1)/4)) ** 2)
    sum_part = np.sum(((x[:-1] + 1)/4)**2 * (1 + 10*(np.sin(np.pi*(1 + (x[1:] + 1)/4))**2)))
    term3 = ((x[-1] + 1)/4)**2
    total = (np.pi / dim) * (term1 + sum_part + term3) + Ufun(x, 10, 100, 4)
    return total

def F18(x):
    dim = len(x)
    term1 = (np.sin(3 * np.pi * x[0]))**2
    sum_part = np.sum((x[:-1] - 1)**2 * (1 + (np.sin(3 * np.pi * x[1:]))**2))
    term3 = (x[-1] - 1)**2 * (1 + (np.sin(2 * np.pi * x[-1]))**2)
    return 0.1 * (term1 + sum_part + term3) + Ufun(x, 5, 100, 4)

def F19(x):
    aS = np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
                   [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
    bS = np.sum((x.reshape(-1,1) - aS)**6, axis=0)
    return 1 / (1/500 + np.sum(1 / (np.arange(1, 26) + bS)))

def F20(x):
    aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    bK = 1 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    return np.sum((aK - (x[0] * (bK**2 + x[1]*bK) / (bK**2 + x[2]*bK + x[3]))**2))

def F21(x):
    return 4*x[0]**2 - 2.1*x[0]**4 + x[0]**6/3 + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4

def F22(x):
    aSH = np.array([[4,4,4,4], [1,1,1,1], [8,8,8,8], [6,6,6,6], [3,7,3,7],
                   [2,9,2,9], [5,5,3,3], [8,1,8,1], [6,2,6,2], [7,3.6,7,3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    o = 0
    for i in range(5):
        diff = x - aSH[i]
        o -= 1 / (np.dot(diff, diff) + cSH[i])
    return o

def F23(x):
    aSH = np.array([[4,4,4,4], [1,1,1,1], [8,8,8,8], [6,6,6,6], [3,7,3,7],
                   [2,9,2,9], [5,5,3,3], [8,1,8,1], [6,2,6,2], [7,3.6,7,3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    o = 0
    for i in range(7):
        diff = x - aSH[i]
        o -= 1 / (np.dot(diff, diff) + cSH[i])
    return o

def F24(x):
    aSH = np.array([[4,4,4,4], [1,1,1,1], [8,8,8,8], [6,6,6,6], [3,7,3,7],
                   [2,9,2,9], [5,5,3,3], [8,1,8,1], [6,2,6,2], [7,3.6,7,3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    o = 0
    for i in range(10):
        diff = x - aSH[i]
        o -= 1 / (np.dot(diff, diff) + cSH[i])
    return o

def get_function_details(F):
    func_details = {
        'F1': {'lb': -100, 'ub': 100, 'dim': 30, 'fobj': F1},
        'F2': {'lb': -10, 'ub': 10, 'dim': 30, 'fobj': F2},
        'F3': {'lb': -1, 'ub': 1, 'dim': 30, 'fobj': F3},
        'F4': {'lb': -100, 'ub': 100, 'dim': 30, 'fobj': F4},
        'F5': {'lb': -100, 'ub': 100, 'dim': 30, 'fobj': F5},
        'F6': {'lb': -30, 'ub': 30, 'dim': 30, 'fobj': F6},
        'F7': {'lb': -100, 'ub': 100, 'dim': 30, 'fobj': F7},
        'F8': {'lb': -1.28, 'ub': 1.28, 'dim': 30, 'fobj': F8},
        'F9': {'lb': -5, 'ub': 10, 'dim': 30, 'fobj': F9},
        'F10': {'lb': -500, 'ub': 500, 'dim': 30, 'fobj': F10},
        'F11': {'lb': -10, 'ub': 10, 'dim': 30, 'fobj': F11},
        'F12': {'lb': -5, 'ub': 5, 'dim': 30, 'fobj': F12},
        'F13': {'lb': -5.12, 'ub': 5.12, 'dim': 30, 'fobj': F13},
        'F14': {'lb': -32, 'ub': 32, 'dim': 30, 'fobj': F14},
        'F15': {'lb': -600, 'ub': 600, 'dim': 30, 'fobj': F15},
        'F16': {'lb': -10, 'ub': 10, 'dim': 30, 'fobj': F16},
        'F17': {'lb': -50, 'ub': 50, 'dim': 30, 'fobj': F17},
        'F18': {'lb': -50, 'ub': 50, 'dim': 30, 'fobj': F18},
        'F19': {'lb': -65, 'ub': 65, 'dim': 2, 'fobj': F19},
        'F20': {'lb': -5, 'ub': 5, 'dim': 4, 'fobj': F20},
        'F21': {'lb': -5, 'ub': 5, 'dim': 2, 'fobj': F21},
        'F22': {'lb': 0, 'ub': 10, 'dim': 4, 'fobj': F22},
        'F23': {'lb': 0, 'ub': 10, 'dim': 4, 'fobj': F23},
        'F24': {'lb': 0, 'ub': 10, 'dim': 4, 'fobj': F24}
    }
    detail = func_details.get(F, None)
    if detail:
        return detail['lb'], detail['ub'], detail['dim'], detail['fobj']
    else:
        raise ValueError("Function not found")