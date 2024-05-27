#include <iostream>
#include <bits/stdc++.h>
using namespace std;

// Q1

//Q2

//Q3 - Find good count
// odd places take prime nums and even places take even nums - given num of digits - how many such nums possible
//ex- n = 5: 3 even and 2 odd = 5*(20)^2 - return count%10e7
// time: O(logn) num of steps red to half each time, space: O(logn)because a function call consumes memory and log N recursive function calls are made
// iter better no usage of memory
class Solution {
public:
    const int MOD = 1000000007;

    long long power(long long x,long long y) {
        if(y == 0 || x == 1)return 1;
        if(x == 0) return 0;

        if((y & 1) == 0) return power(x*x%MOD, y/2)%MOD;
        else return x%MOD * power(x*x%MOD, (y-1)/2)%MOD ;
    }

    int countGoodNumbers(long long n) {       
        long long result = power(20, n/2);
        if(n & 1 == 1)result = result*5 % MOD;
        return result;
    }
};

// less mem usage method
#define ll long long

class Solution {
public:
    // evens  = 0, 2, 4, 6, 8  => 5 evens
    // primes = 2, 3, 5, 7     => 4 primes
  
    int p = 1e9 + 7;
    
    // calculating x^y efficeiently
    ll power(long long x, long long y) {
      long long res = 1;    

      x = x % p; // Update x if it is more than or equal to p
      if (x == 0) return 0; 

      while (y > 0) {
        // If y is odd, multiply x with result
        if (y & 1) res = (res*x) % p;
        
        // we have did the odd step is it was odd, now do the even step
        y = y>>1; // dividing y by 2, y>>1 is same as y/2
        x = (x*x) % p;
      }
      return res;
    }
  
    int countGoodNumbers(long long n) {
      ll count_of_4s = n/2;
      ll count_of_5s = n - count_of_4s;
      ll ans = ((power(4LL, count_of_4s) % p  * power(5LL, count_of_5s) % p) % p);
      return (int)ans;
    }
};
