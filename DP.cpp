#include <iostream>
#include <bits/stdc++.h> 
using namespace std;

// 1 1D DP-------------------------------------------
// Q1

// Q2 - Frog jump
// frog can jump from i to i+1 or i+2th step, cost per jump i to j = abs(height(i)-height(j))
// 1D dp- time: O(n), space: O(n)
int minimumEnergy(vector<int>& height, int n) {
    if(n == 1)return 0;
    if(n == 2)return height[1]-height[0];
    vector<int>dp(n,0);
    dp[1] = abs(height[1]-height[0]);
    for(int i = 2; i < n; i++){
        dp[i] = min(dp[i-1]+abs(height[i-1]-height[i]), dp[i-2]+abs(height[i-2]-height[i]));
    }
    return dp[n-1];
}

// Q3 - Frog Jump with k distances
// same as the above q
// failed for some reason, i think the test case format is incorrect
int minimizeCost(vector<int>& height, int n, int k) {
    if(n <= k)return abs(height[k]-height[0]);
    vector<int>dp(n,0);
    for(int i = 0; i < k+1; i++){
        dp[i] = abs(height[i]-height[0]);
    }
    
    for(int i = k+1; i < n; i++){
        int m = 1e4 + 1;
        for(int j = i-1; j >= i-k; j--){
            int cost = dp[j]+abs(height[j]-height[i]);
            if(cost < m)m = cost;
        }
        if(dp[i] < 0)cout<<
        dp[i] = m;
    }
    return dp[n-1];
}

// Q4 
// dp[i] = max amount till house i (may or may not rob ith house)
// dp[i] = either the same as dp[i-1](not take ith house) or dp[i-2]+nums[i](take ith house)
// time: O(n)-100%, space: O(n) 10%-can be optimised
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n,0);
        dp[0] = nums[0];
        if(n>1)dp[1] = max(nums[0], nums[1]);
        for(int i = 2; i < n;i++){
            dp[i] = max(dp[i-1], dp[i-2]+nums[i]);
        }
        return dp[n-1];
    }
};

// another method
// Variable DP approach - without dp table - can use pts to track ans for i-1 and i-2
// prev1 - ans till i-1th house, prev2 is ans till i-2th house
// time: O(n)-100%, space: O(1)-80%
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        if(n == 1)return nums[0];
        if(n == 2)return max(nums[0], nums[1]);
        int prev2 = nums[0], prev1 = max(nums[0], nums[1]);
        int temp = 0;
        for(int i = 2; i < n; i++){
            temp = prev1;
            prev1 = max(temp, prev2+nums[i]);
            prev2 = temp;
        }
        return prev1;
    }
};

// Q5 - same q as above but houses are in a circle => 1st and last house cant be robbed together

// brute for recursion - take, not take time:O(2^n), auxillary space:O(n)

// Memoization - time:O(n), space:O(2n)
class Solution {
private:
    int dp[101][2];
    int getMax(vector<int>&nums, int i, bool robFirst){
        if(i >= nums.size() || (i==nums.size()-1 && robFirst)) return 0;
        if(dp[i][robFirst]!=-1) return dp[i][robFirst];
        int rob = 0, notRob = 0; 
        if(i==0) rob = getMax(nums,i+2,1)+nums[i]; 
        else rob = getMax(nums,i+2,robFirst)+nums[i];
        notRob = getMax(nums,i+1,robFirst);
        return dp[i][robFirst] = max(rob, notRob);
    }
public:
    int rob(vector<int>& nums) {
        memset(dp, -1, sizeof dp);
        return getMax(nums, 0, 0);
    }
};

// TABULATION - time:O(n), space:O(2n)
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        int dp[n+1], dp1[n+1];
        dp[0] = 0 ; 
        dp[1] = nums[0];
        dp1[0] = 0;
        dp1[1] = 0;
        for(int i = 2 ; i <= n ; i++){
            if(i == n) dp[i] = dp[i-1];
            else dp[i] = max(dp[i-1], dp[i-2]+nums[i-1]);
            dp1[i] = max(dp1[i-1], dp1[i-2]+nums[i-1]);
        }  
        return max(dp[n], dp1[n]);
    }
};

// Variable DP
// take 1 - (q4 for nums[0...n-1]), not take 1 - (q4 for nums[1...n])
// time:O(n) - 100%, space:O(1) - 70%
class Solution {
public:
    int helper(vector<int>& nums, int start){
        int prev2 = nums[start], prev1 = max(nums[start], nums[start+1]);
        int temp = 0;
        for(int i = start+2; i < nums.size()-1+start; i++){
            temp = prev1;
            cout<<i;
            prev1 = max(temp, prev2+nums[i]);
            prev2 = temp;
        }
        return prev1;
    }


    int rob(vector<int>& nums) {
        int n = nums.size();
        if(n == 1)return nums[0];
        if(n == 2)return max(nums[0], nums[1]);
        if(n == 3)return max(max(nums[0], nums[1]),nums[2]);
        return max(helper(nums, 0), helper(nums,1));
    }
};

// 2- 2D,3D DP and DP on grids
// Q1
// 2D tabulation
// space: O(3n)
int maximumPoints(vector<vector<int>>& points, int n) {
    vector<vector<int>> dp(n, vector<int>(3, 0));
    dp[0][0] = points[0][0]; dp[0][1] = points[0][1]; dp[0][2] = points[0][2];
    
    for (int i = 1; i < n; ++i) {
        dp[i][0] = points[i][0] + max(dp[i-1][1], dp[i-1][2]);
        dp[i][1] = points[i][1] + max(dp[i-1][0], dp[i-1][2]);
        dp[i][2] = points[i][2] + max(dp[i-1][0], dp[i-1][1]);
    }
    
    return max(dp[n-1][0], max(dp[n-1][1], dp[n-1][2]));
}

// space: O(n), 1D array
int maximumPoints(vector<vector<int>>& points, int n) {
    vector<int> dp(n);

    dp[0] = max(points[0][0], max(points[0][1], points[0][2]));

    vector<int> prev(3);
    prev[0] = points[0][0];
    prev[1] = points[0][1];
    prev[2] = points[0][2];

    for (int i = 1; i < n; ++i) {
        vector<int> curr(3);
        curr[0] = points[i][0] + max(prev[1], prev[2]);
        curr[1] = points[i][1] + max(prev[0], prev[2]);
        curr[2] = points[i][2] + max(prev[0], prev[1]);

        dp[i] = max(curr[0], max(curr[1], curr[2]));

        prev = curr;
    }

    return dp[n-1];
}

// space: O(1)
int maximumPoints(vector<vector<int>>& points, int n) {
    vector<int> dp(3, 0);

    dp[0] = points[0][0];
    dp[1] = points[0][1];
    dp[2] = points[0][2];

    // Temporary array to store current day's maximum points
    vector<int> current(3, 0);

    // Fill the dp array for each day
    for (int i = 1; i < n; ++i) {
        current[0] = points[i][0] + max(dp[1], dp[2]);
        current[1] = points[i][1] + max(dp[0], dp[2]);
        current[2] = points[i][2] + max(dp[0], dp[1]);
        
        // Update dp for the next iteration
        dp = current;
    }

    return max(dp[0], max(dp[1], dp[2]));
}

