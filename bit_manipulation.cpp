#include <iostream>
#include <bits/stdc++.h>
using namepace std;

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