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
            r++;
            if(maxlen < len)maxlen = len;
        }
        return maxlen;
    }
};

//converting the as the largest subarr with atmost k zeroes
// better sol'n - time complexity
// time: O(2n), apce: O(1) 
class Solution {
public:
    int longestOnes(vector<int>& A, int K) {
       int start = 0, ans = INT_MIN,zeroCounter=0;
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
            ans = max(ans,end-start+1);
        } 
        return ans;
    }
};

// optimal : time: O(n), space: O(1)
// because of while - len dec as l inc and r is same, instead we can inc both l and r => sliding window
class Solution {
public:
    int longestOnes(vector<int>& A, int K) {
       int start = 0, ans = INT_MIN,zeroCounter=0;
        for(int end = 0 ; end < A.size() ; end++){
            if(A[end] == 0){
                zeroCounter++;
            }
            if(zeroCounter > K){
                if(A[start]==0){
                    zeroCounter--;
                }
                start++;
            }
            if(zeroCounter <= K) ans = max(ans,end-start+1);
        }
        return ans;
    }
};

// Q3 - fruits into baskets
// you have 2 baskets and each backet can only contain a fruit of the same type
// whats the max of of fruits you can take ex: 2 3 1 2 - ans = 2 (1,2)

// method 1: time: O(2*n) space: O(3)
// ptr r adds an element and if we encounter a fruit of new type - start removing fruit to the left, each iter record the max len
int totalFruits(int N, vector<int> &arr) {
    int l = 0, r = 0, ans = 0;
    unordered_map <int,int> mp;
    for(int r = 0; r < N; r++){
        mp[arr[r]]++;
        while(mp.size() > 2){
            if(mp[arr[l]] == 0 || mp[arr[l]] == 1) mp.erase(arr[l]);
            else mp[arr[l]]--;
            l++;
        }
        ans = max(ans, r-l+1);
    }
    return ans;
}

// method 2: time: O(n), space: O(3)
// to reduce time, once we encounter 3 different types of fruits, we slide l and r i.e len never reduceseither maintains or increases)

int totalFruits(int N, vector<int> &arr) {
    int l = 0, ans = 0;
    unordered_map <int,int> mp;
    for(int r = 0; r < N; r++){
        mp[arr[r]]++;
        if(mp.size() > 2){
            mp[arr[l]]--;
            if(mp[arr[l]] == 0) mp.erase(arr[l]);
            l++;
        }
        if(mp.size() <= 2) {ans = max(ans, r-l+1);
        }
    }
    return ans;
}

// Q4 longest repeating char replacement
// time: O(n), space: O(n)
class Solution {
public:
    int characterReplacement(string s, int k) {
        int l = 0, ans = 0, sum = 0;
        char c = s[0];
        unordered_map<int, int> mp;
        for (int r = 0; r < s.size(); r++) {
            mp[s[r]]++;
            if (s[r] != c && mp[s[r]] > mp[c]) {
                sum -= (mp[s[r]] - 1);
                sum += mp[c];
                c = s[r];
            } else if (s[r] != c && mp[s[r]] <= mp[c])
                sum++;

            if (sum > k) {
                mp[s[l]]--;
                if (mp[s[l]] == 0)
                    mp.erase(s[l]);
                if (s[l] != c)
                    sum--;
                l++;
            }
            if (sum <= k)
                ans = max(ans, r - l + 1);
        }
        return ans;
    }
};

// Q5 - binary subarray with sum
class Solution {
public:
    int numSubarraysWithSum(vector<int>& nums, int goal) {
        return atMost(nums, goal) - atMost(nums, goal - 1);
    }
    
private:
    int atMost(vector<int>& nums, int goal) {
        if (goal < 0) return 0;
        int n = nums.size();
        int l = 0, r = 0, sum = 0, cnt = 0;
        while (r < n) {
            sum += nums[r];
            while (sum > goal) {
                sum -= nums[l];
                l++;
            }
            cnt += r - l + 1;
            r++;
        }
        return cnt;
    }
};
