#include <iostream>
#include <bits/stdc++.h> 
using namespace std;

// Q2 max consecutive ones 
// k o's can be flipped and counted
// my sol: 

class Solution {
public:
    int longestOnes(vector<int>& nums, int k) {
        int l = 0, r = 0;
        int len = 0, maxlen = 0, flips = 0;
        int n= nums.size();
        
        while(r < n){
            if(nums[r] == 1) len++;
            else {
                if(flips == k) {
                    if(nums[l] == 1){
                        do{ l++; len--;}
                        while(nums[l] != 0);
                    }
                    l++;
                }
                else {flips++;len++;}
            }
            cout<<l<<" "<<r<<" "<<len<<" "<<flips<<endl;
            r++;
            if(maxlen < len)maxlen = len;
        }
        return maxlen;
    }
};

// better sol'n - time complexity

class Solution {
public:
    int longestOnes(vector<int>& A, int K) {
       int start = 0, max = INT_MIN,zeroCounter=0;
        for(int end = 0 ; end < A.size() ; end++){
            if(A[end] == 0){
                zeroCounter++;
            }
            while(zeroCounter > K){
                if(A[start]==0){
                    zeroCounter--;
                }
                start++;
            }
            
            max = max(max,end-start+1);
        }
        
        return max;
    }
};