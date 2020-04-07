from itertools import combinations_with_replacement


def grid_weights_gen(w_size=4, grid_size=16):
    a = list(range(grid_size+1))
    combi = combinations_with_replacement(a, w_size-1)
    block_size = 1 / grid_size
    result = []
    for c in combi:
        assigned = 0
        temp = []
        for bar in c:
            num = bar - assigned
            w = num * block_size
            temp.append(w)
            assigned += num
        num = grid_size - assigned
        w = num * block_size
        temp.append(w)
        result.append(temp)
            
    return result



if __name__ == "__main__":
    li = grid_weights_gen()
    for i in range(len(li)):
        print(li[i])
    print(len(li))
