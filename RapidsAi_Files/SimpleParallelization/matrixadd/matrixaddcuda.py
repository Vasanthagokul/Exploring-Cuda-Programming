# data = [2,4,5,7,8,9,12,14,17,19,22,25,27,28,33,37]
# target = 28

# #Linear Search
# def linear_search(data, target):
#     for i in range(len(data)):
#         if data[i] == target:
#             return True
#     return False

# # Iterative Binary Search
# def binary_search_iterative(data,target):
#     low = 0
#     high = len(data) - 1

#     while low <= high:
#         mid = (low + high) // 2
#         if target == data[mid]:
#             return True
#         elif target < data[mid]:
#             high = mid - 1
#         else:
#             low = mid
#     return False
import numpy as np
import time

from numba import vectorize, cuda

@vectorize(['float32(float32, float32)'], target='cuda')
def VectorAdd(a, b):
    return a + b

def main():
    N = 32000000

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    start = time.time()
    C = VectorAdd(A, B)
    vector_add_time = time.time() - start
 
    print(f"C[:5] = {str(C[:5])}")
    print(f"C[-5:] = {str(C[-5:])}")

    print(f"VectorAdd took for % seconds {vector_add_time}")

if __name__=='__main__':
    main()