#include <iostream>
#include <bits/stdc++.h>
using namespace std;

// basics - brute force - conveert to binary
//swapping with no temp var - using xor
// XOR - even no. of 1's - 0
int a , b;
a = a^b;
b = a^b; // (a^b)^b = a (now b = a)
a = a^b; // (a^b)^a = (a^b)^a = b

// check ith bit
// with <<:  num & (1 << i) 
// with >>:  (num >> i) & 1 

// set ith bit: num | (1 << i)
// clear ith bit: num & ~(1 << i)
// toggle ith bit: num ^ (1 << i)
// remove last set bit: from observation : num & (num-1)
// if n has i as the rightmost set bit then (n-1) has 1's till i-1
// check if a num is pow of 2: num & (num-1) == 0
// count num of set bits :
// bit wise operators are faster - O(n)
int count_set_bits(int n){
    int cnt = 0;
    while(n > 1){
        cnt += n&1;
        n  = n>>2;
    }
    if(n == 1)cnt++;
    return cnt;
}

// better method - O(set_bits)<= O(31)
int count_set_bits(int n){
    int cnt = 0;
    while(n != 0){
        n = n&(n-1);
        cnt++;
    }
    return cnt;
}

// Q1 check, set and clear ith bit
void bitManipulation(int num, int i) {
        int k = num;
        k = k >> (i-1);
        int get = k%2; //k&1
        int set = num, clear = num;
        if(get == 1) clear = num - pow(2,i-1);
        else set = num + pow(2,i-1);
        cout << get << " " << set << " " << clear;
}

// Q8 divide 2 nums return integer part
class Solution {
public:
    int divide(int dividend, int divisor) {
        if(dividend == 0)return 0;
        if(dividend == divisor)return 1;
        bool sign = true;
        if(dividend >= 0 && divisor < 0)sign = false;
        else if(dividend <= 0 && divisor > 0)sign = false;
        long n = abs(dividend);
        long p = abs(divisor);
        long q = 0;
        while(n >= p){
            int cnt = 0;
            while(n >= (p << (cnt+1)))cnt++;
            q += 1<<cnt;
            n -= (p << cnt);
        }
        if(q == (1<<31) && sign)return INT_MAX;
        if(q == (1<<31) && !sign)return INT_MIN;
        return sign?q:-q;
    }
};

// -------------------INTERVIEW--------------------------------------------
// Q1
// num of bits needed to be flipped to convet start to goal
// check the once digit each time and do right shift
// time: O(log(n+m)), space: O(1)
class Solution {
public:
    int minBitFlips(int start, int goal) {
        int ans = 0;
        while(start != 0 || goal != 0){
            int k = ((start & 1) ^ (goal & 1));
            if(k == 1) ans++;
            goal = (goal >> 1);
            start = (start >> 1); 
        }  
        return ans;  
    }
};

// Q2 find the only num which appears odd num of time (all other nums in arr appear even num of times)
// xor
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ans = 0;
        for(int i = 0; i < nums.size(); i++){
            ans = ans ^ nums[i];
        }
        return ans;
    }
};

// Q3 subset sum


//Q4 xor l to r
// given [L,R] inclusive find xor from l to r
// in O(1) time and space
// mathematical - find pattern; idea: even repeat of nums when xored become 0
class Solution {
  public:
    int xor12n(int i){
        if(i%4 == 1) return 1;
        if(i%4 == 2) return i+1;
        if(i%4 == 3) return 0;
        else return i;
    }
  
    int findXOR(int l, int r) {
        return (xor12n(l-1) ^ xor12n(r));
    }
};


