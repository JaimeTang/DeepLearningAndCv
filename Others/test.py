import numpy as np


class Solution:
    """
    @param nums: The integer array you should partition
    @param k: An integer
    @return: The index after partition
    """
    def partitionArray(self, nums, k):
        # write your code here
        left = 0
        length = len(nums)
        nums.sort()
        while left < length:
            if nums[left] < k:
                left += 1
            else:
                break
        return left




def main():
    list = [[1,2,3,6],[5,5,7,3],[5,5,7,8]]
    list = np.array(list)
    score = list[:, 3]
    print(score.argsort()[::-1])


if __name__=='__main__':
    main()