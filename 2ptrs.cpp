#include <iostream>
#include <bits/stdc++.h> 
using namespace std;

// for longest subarr you need to slide (keeping the window size same, cuz len req is max - either remain smae/increases)
// for the num of subarr type- the size of window may change so we convert the qurstion to func(nums,cond)-func(nums,smaller cond)=func(nums,goal)


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

// Q5 - count binary subarray with sum
// since we need to count- our window may inc/dec in length -> O(n^2)
// we can try to reframe the question as =>( the num of subarr with sum <= goal) - (num of subarr with sum <= goal-1) == num of subarr with sum = goal
// we are doing this because of zeroes, it its anyother num this is not needed - zero doesn't affect/change the given cond on its own
// time:O(n), space:O(1)
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

// Q6 count num of subarr with k odd nums
// similar to above q - even nums do not affect the cond
class Solution {
public:
    int numberOfSubarrays(vector<int>& nums, int k) {
        return atMost(nums, k) - atMost(nums,k-1);
    }

private: 
    int atMost(vector<int>& nums, int k){
        if(k < 0)return 0;
        int l = 0, ans = 0, cnt = 0;
        for(int r = 0; r < nums.size(); r++){
            if((nums[r]&1) == 1) cnt++;
            while(cnt > k){
                if((nums[l]&1) == 1)cnt--;
                l++;
            }
            ans+=r-l+1;
            // cout<<l<<" "<<r<<" "<<r-l+1<<" "<<ans<<endl;
        }
        return ans;
    }
};

// Q7 count number of substr containing all 3 chars a,b,c
// a different approach - from the above methods - find the smallest window possible ending at i, then add all the prev idx => all the subarr ending at i with all 3 chars
// time: O(n), space: O(1)
class Solution {
public:
    int numberOfSubstrings(string s) {
        int a = -1, b = -1, c = -1, ans = 0;
        for(int i = 0; i < s.size(); i++){
            if(s[i] == 'a') a = i;
            else if(s[i] == 'b') b = i;
            else if(s[i] == 'c') c = i;
            if(a != -1 && b != -1 && c != -1){
                ans += (min(min(a,b),c)+1);
            }
        }
        return ans;
    }
};

// Q8 max points you can obtain from choosing a card from left/right end
// brute: either recursion i.e max( func(arr[l...r-1])+arr[r] , arr[l]+func(arr[l+1...r])) - O(2^n)
class Solution {
public:
    int maxScore(vector<int>& cardPoints, int k) {
        return helper(cardPoints, 0, cardPoints.size()-1, k);
    }

private:
    int helper(vector<int>& cardPoints, int l, int r, int k){
        if(k == 0)return 0;
        int left = helper(cardPoints, l+1, r, k-1) + cardPoints[l];
        int right = helper(cardPoints, l, r-1, k-1) + cardPoints[r];
        return max(left,right);
    }
};

// brute: time:O(2k), space: O(2k)
// store the sum of arr[0...k] subarr till i, and sum of arr[n-k-1...n-1] from end
// i.e for i ele from left k-i ele from right selected - so ans is i <- 0 to k : max(left_sum[i]+right_sum[i])
class Solution {
public:
    int maxScore(vector<int>& cardPoints, int k) {
        vector<int>left(k+1, 0);
        vector<int>right(k+1, 0);
        int n = cardPoints.size();
        for(int i = 0; i < k; i++){
            left[i+1] = left[i] + cardPoints[i];
            right[i+1] = right[i] + cardPoints[n-1-i];
            cout<<left[i+1] << " "<<right[i+1]<<endl;
        }
        int ans = 0;
        for(int i = 0; i < k+1; i++){
            ans = max(ans, left[i]+right[k-i]);
        }
        return ans;
    }
};


// optimal time:O(2k), space: O(1)
// keeping track of max insead of using extra space
class Solution {
public:
    int maxScore(vector<int>& cardPoints, int k) {
        int l = 0, r = 0, ans = 0, n = cardPoints.size();
        int l_sum = 0, r_sum = 0;
        for(int i = 0; i < k; i++){
            l_sum += cardPoints[i];
            r_sum += cardPoints[n-1-i];
        }
        for(int i = 0; i < k; i++){
            ans = max(max(r+l_sum, l+r_sum),ans);
            l += cardPoints[i]; r += cardPoints[n-1-i];
            r_sum -= cardPoints[n-k+i]; l_sum -= cardPoints[k-1-i];
        }
        return ans;
    }
};


// -------------------HARD---------------------------------------------
// Q3 - the smallest window(substr) of str s containing all elements of str t (includeing dup)
// brute: O(n^2) for every char in s as starting check if all ele of t are present - use hashmap to store ele of t 
// better: time: O(2n), space: O(1)
// using 2 ptr., ptr r will reduce the ele from hashmap (+ve in map rep: ele in t not yet seen in s, -ve rep ele in s seen but not present in t)
// for any pos ele in hashmap found - cnt++
// cnt represents that we have found an ele present in t in s(not represted in t but duplicates say- t-'aba' and s-'aaba')then at i=2 cnt = 2(dup counted) if t - 'ab' s-'aaba' then at i = 1, cnt = 1(not 2) 
// this is ensured because cnt is only inc if it is already +ve in hashmap (ele in t not yet seen in s), once cnt == t.size()- all ele in t have been seen only then count can inc

class Solution {
public:
    string minWindow(string s, string t) {
        if (t.size() > s.size()) return "";
        
        unordered_map<char, int> mp;
        for (char c : t) mp[c]++;
        
        int l = 0, r = 0, minlen = INT_MAX, start = -1, cnt = 0;
        
        for(int r = 0; r < s.size(); r++) {
            if (mp[s[r]] > 0) cnt++;
            mp[s[r]]--;
            
            while (cnt == t.size()) {
                if (r - l + 1 < minlen) {
                    minlen = r - l + 1;
                    start = l;
                }
                mp[s[l]]++;
                if (mp[s[l]] > 0) cnt--;
                l++;
            }
        }

        return (start == -1) ? "" : s.substr(start, minlen);
    }
};