# 191. Number of 1 Bits https://leetcode.com/problems/number-of-1-bits/
class Solution:
    def hammingWeight(self, n: int) -> int:
        count=0
        while(n!=0):
            n=n&(n-1)
            count+=1
        return count

# 50. Pow(x, n) https://leetcode.com/problems/powx-n/
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n<=0:
            return 1/self.helper(x,-n)
        return self.helper(x,n)
    
    def helper(self,x:float,n:int)->float:
        pow=1
        while n:
            if n&1:
                pow*=x
            x*=x
            n>>=1
        return pow

# 位运算：https://leetcode.com/problems/single-number-iii/description/
# Input:  [1,2,1,3,2,5]
# Output: [3,5]
# 首先先用异或得到两个数字的异或值
# 比如出现一次的两个数字是3(001)，5(110),r异或值(101)
# r & ~(r-1) 用来截取最后的最开始的一位不一样的数字，比如 r=101 ,r & ~(r-1)=1,可以作为mask分割两个数字
# r & (r-1) 消除最后一个1,比如 r=101 ,r & (r-1)=100,
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        r = 0
        for n in nums:
            r ^= n
        # 1 to the right most
        mask = r & ~(r-1)
        res = [0, 0]
        for n in nums:
            if n & mask:
                res[0] ^= n
            else:
                res[1] ^= n
        return res

# 405. Convert a Number to Hexadecimal https://leetcode.com/problems/convert-a-number-to-hexadecimal/
class Solution:
    def toHex(self, num: int) -> str:
        if num == 0:
            return "0"
        mp = "0123456789abcdef"
        rs = ""
        for i in range(8):
            rs = mp[num % 16] + rs
            num = num >> 4
        return rs.lstrip('0')

# 66. Plus One https://leetcode.com/problems/plus-one/
# Input: [1,2,3]
# Output: [1,2,4]
# Explanation: The array represents the integer 123.
# 妙哉
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        if len(digits)==0:
            return False
        addCarry=1
        for i in range(len(digits)-1,-1,-1):
            digits[i]+=addCarry
            if digits[i]==10:
                digits[i]=0
                if i==0:
                    digits.insert(0,1)
            else:
                break
        return digits

# 进制转换：https://leetcode.com/problems/base-7/description/
class Solution:
    def convertToBase7(self, num: int) -> str:
          ans = ""
          if num == 0:
              return '0'
          x = abs(num)
          while(x > 0):
              ans += str(x%7)
              x //= 7
          if (num > 0):
              return ans[::-1]
          else:
              return '-' + ans[::-1]