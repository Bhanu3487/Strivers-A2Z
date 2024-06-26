#include <iostream>
#include <bits/stdc++.h> 
using namespace std;

// DP isn't just storing values in arrays, it consists of breaking down a bigger problem into many smaller problems

// ex. f(n) = f(n-2)+f(n-5) and n = 6
// tabulation - fill the array completely - need to predict cond - faster
// we find for all n from 1 to 6
// from req to base cond. back to req cond.
// memoization - we only fill what we want - easier logic - recursion, slow due to recurive calls
// we find for 6 after 4 and 1, after 2  
// from base case to req cond.


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

// Q3 - grid with obstacles
// 2D DP time:O(mn) space:O(mn)
int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
    int n = obstacleGrid.size(), m = obstacleGrid[0].size();
    vector<vector<int>>dp (n, vector<int>(m,0));

    dp[0][0] = 1;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            if(obstacleGrid[i][j] == 0){
                if(i < n-1 && obstacleGrid[i+1][j] == 0){dp[i+1][j] += dp[i][j];} //right
                if(j < m-1 && obstacleGrid[i][j+1] == 0){dp[i][j+1] += dp[i][j];} //down
            }
            else dp[i][j] = 0;
        }
    }

    return dp[n-1][m-1];
}

// time: O(mn), space:O(1)
int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
    int h = obstacleGrid.size();
    if(h == 0) return 0;
    int w = obstacleGrid[0].size();
    if(w == 0) return 0;
    if(obstacleGrid[0][0]) return 0;
    
    // first cell has 1 path
    obstacleGrid[0][0] = 1;
    
    // first row all are '1' until obstacle (from left only)
    for(int i=1; i<w; i++){
        obstacleGrid[0][i] = obstacleGrid[0][i] ? 0 : obstacleGrid[0][i-1];
    }

    for(int j=1; j<h; j++){
        // first column is like first row (from top only)
        obstacleGrid[j][0] = obstacleGrid[j][0] ? 0 : obstacleGrid[j-1][0];
        
        // others are up+left
        for(int i=1; i<w; i++){
            obstacleGrid[j][i] = obstacleGrid[j][i] ? 0 : obstacleGrid[j-1][i] + obstacleGrid[j][i-1];
        }
    }
    
    return obstacleGrid[h-1][w-1];
}

// Q3 - grid path with least weight
// 2D DP in place; time:O(mn), space:O(1)
int minPathSum(vector<vector<int>>& grid) {
    int n = grid.size(), m = grid[0].size();

    for(int j = 1; j < m; j++){
        grid[0][j] += grid[0][j-1];
    }    

    for(int i = 1; i < n; i++){
        grid[i][0] += grid[i-1][0];

        for(int j = 1; j < m; j++){
            grid[i][j] += min(grid[i][j-1],grid[i-1][j]);
        }  
    }
    return grid[n-1][m-1];
}

// Q4
// a lower triangular grid - can move from (i,j) to (i+1,j) and (i+1,j+1)- find the path with least weight
// https://leetcode.com/problems/triangle/solutions/2146264/c-python-simple-solution-w-explanation-recursion-dp/ - all approaches- recursion, Top-Down DP or Memoization, Bottom Up DP or Tabulation, Bottom Up DP or Tabulation (Space Optimized)(storing the last row in prev row)

// 2D DP Memoization, time:O(mn), space:O(mn)
class Solution {
   public:
    int dfs(int i, int j, int n, vector<vector<int>>& triangle, vector<vector<int>>& memo) {
        if (i == n) return 0;
        if (memo[i][j] != -1) return memo[i][j];
        
        int lower_left = triangle[i][j] + dfs(i + 1, j, n, triangle, memo);
        int lower_right = triangle[i][j] + dfs(i + 1, j + 1, n, triangle, memo);
        memo[i][j] = min(lower_left, lower_right);
        
        return memo[i][j];
    }
    int minimumTotal(vector<vector<int>>& triangle) {
        int n = triangle.size();
        vector<vector<int>> memo(n, vector<int>(n, -1));
        return dfs(0, 0, n, triangle, memo);
    }
};

// Tabulation space:O(n^2)
int minimumTotal(vector<vector<int>>& triangle) {
    int n = triangle.size();
    vector<vector<int>> dp(n, vector<int>(n, -1));
    for (int j = 0; j < n; j++) dp[n - 1][j] = triangle[n - 1][j];
    for (int i = n - 2; i >= 0; i--) {
        for (int j = 0; j < i + 1; j++) {
            int lower_left = triangle[i][j] + dp[i + 1][j];
            int lower_right = triangle[i][j] + dp[i + 1][j + 1];
            dp[i][j] = min(lower_left, lower_right);
        }
    }
    return dp[0][0];
}


// Tabuation space:O(n)
int minimumTotal(vector<vector<int>>& triangle) {
    int n = triangle.size();
    vector<int> next_row(triangle[n - 1]);
    vector<int> curr_row(n, 0);
    for (int i = n - 2; i >= 0; i--) {
        for (int j = 0; j < i + 1; j++) {
            int lower_left = triangle[i][j] + next_row[j];
            int lower_right = triangle[i][j] + next_row[j + 1];
            curr_row[j] = min(lower_left, lower_right);
        }
        swap(curr_row, next_row);
    }
    return next_row[0];  // because we swapped at last iteration
}

// space:O(1) - array
int minimumTotal(vector<vector<int>>& triangle) {
    int n = triangle.size();
    for(int i = 1; i < n; i++){
        triangle[i][0] += triangle[i-1][0];
        triangle[i][i] += triangle[i-1][i-1];
    }

    for(int i = 2; i < n; i++){
        for(int j = 1; j < i; j++){
            triangle[i][j] += min(triangle[i-1][j], triangle[i-1][j-1]);
        }
    }
    
    int ans = INT_MAX;
    for(int j = 0; j <= n-1; j++){
        ans = min(ans, triangle[n-1][j]);
    }
    return ans;
}

// Q5 
// a lower triangular grid - can move from (i,j) to (i+1,j) and (i+1,j+1)- find the path with least weight

// 2D Top-Down Memoization - time:O(n^2), space:O(n^2)
class Solution {
public:
    int dfs(int i, int j, vector<vector<int>>& matrix, vector<vector<int>>& memo){
        if(memo[i][j] != INT_MAX)return memo[i][j];

        int left = (j != 0)? dfs(i+1, j-1, matrix, memo):INT_MAX;
        int down = dfs(i+1, j, matrix, memo);
        int right = (j != matrix.size()-1)? dfs(i+1, j+1, matrix, memo):INT_MAX;
        memo[i][j] = matrix[i][j] + min(left,min(down,right));
        return memo[i][j];
    }

    int minFallingPathSum(vector<vector<int>>& matrix) {
        int n = matrix.size();
        vector<vector<int>> memo(n, vector<int>(n,INT_MAX));
        for(int j = 0; j < n; j++){
            memo[n-1][j] = matrix[n-1][j];
        }

        int ans = INT_MAX;
        for(int j = 0; j < n; j++){
            ans = min(ans,dfs(0, j, matrix, memo));
        }
        return ans;
    }
};

// 2D Bottom-Up Tabulation - time:O(n^2), space:O(1)
class Solution {
public:
    int minFallingPathSum(vector<vector<int>>& matrix) {
        int n = matrix.size();

        for(int i = n-2; i >= 0; i--){
            for(int j = 0; j < n; j++){
                int left = (j != 0)? matrix[i+1][j-1]:INT_MAX;
                int down = matrix[i+1][j];
                int right = (j != matrix.size()-1)? matrix[i+1][j+1]:INT_MAX;
                matrix[i][j] += min(left,min(down,right));
            }
        }

        int ans = INT_MAX;
        for(int j = 0; j < n; j++){
            ans = min(ans, matrix[0][j]);
        }

        return ans;
    }
};

// same logic different syntax (time: from 28 to 77)
class Solution {
public:
    int customMin(int a, int b, int c){
        return min(a,min(b,c));
    }
    int minFallingPathSum(vector<vector<int>>& matrix) {
        int rows = matrix.size(), cols = matrix[0].size();
        int ans = INT_MAX;
        
        for(int r=1; r < rows; r++){
            for(int c=0; c < cols; c++){
                int leftD, middle, rightD;               
                if(c == 0){
                    rightD = matrix[r-1][c+1];
                    middle = matrix[r-1][c];
                    matrix[r][c] += min(rightD, middle);
                }else if(c == cols-1){
                    leftD = matrix[r-1][c-1];
                    middle = matrix[r-1][c];
                    matrix[r][c] += min(leftD, middle);
                }else{
                    leftD = matrix[r-1][c+1];
                    middle = matrix[r-1][c];
                    rightD = matrix[r-1][c-1];
                    matrix[r][c] += customMin(leftD, rightD, middle);
                } 
            }
        }
        for(int c=0; c < cols; c++){
            ans = min(ans, matrix[rows-1][c]);
        }
        return ans;
    }
};



// DP with Subsequences ---------------------------------------------------------------------------
// Q1 - subsequence sum

// 2D DP Memoization

int func(vector<int>& arr, int idx, int sum, vector<vector<int>>& dp) {
    if (sum == 0) return 1;
    if (idx == 0) return (sum == arr[0]);
    if (dp[idx][sum] != -1) return dp[idx][sum];

    int notTake = func(arr, idx - 1, sum, dp);
    if (notTake == 1) return dp[idx][sum] = 1;

    int take = 0;
    if (sum >= arr[idx]) take = func(arr, idx - 1, sum - arr[idx], dp);

    return dp[idx][sum] = take | notTake;
}

bool isSubsetSum(vector<int>& arr, int sum) {
    vector<vector<int>> dp(arr.size(), vector<int>(sum + 1, -1));
    return func(arr, arr.size() - 1, sum, dp);
}

// 2D DP tabulation
bool isSubsetSum(vector<int>& arr, int sum) {
        vector<vector<bool>> dp(arr.size(), vector<bool>(sum+1, false));
        for(int i = 0; i < arr.size(); i++)dp[i][0] = true;
        dp[0][arr[0]] = true;
        for(int i = 1; i < arr.size(); i++){
            for(int t = 1; t <= sum; t++){
                bool not_take = dp[i-1][t];
                bool take = false;
                if(arr[i] <= t)take = dp[i-1][t-arr[i]];
                dp[i][t] = take | not_take;
            }
        }
        return dp[arr.size()-1][sum];
    }

// 1D DP - Space Optimization
bool isSubsetSum(vector<int>& arr, int sum) {
    int n = arr.size();
    vector<bool> dp(sum + 1, false); // dp[i] will be true if there is a subset of arr that sums up to i
    dp[0] = true; // Base case: subset sum of 0 is always possible (by choosing no elements)
    
    for (int num : arr) {
        for (int s = sum; s >= num; --s) {
            if (dp[s - num]) {
                dp[s] = true;
                // Early termination if we can achieve the sum early
                if (s == sum) return true;
            }
        }
    }
    
    return dp[sum];
}

// Q2 - sum of sunsequences
// Recursion -TLE - not the type of recursion that can be converted to dp bcz it a binary tree -> NO OVERLAPPINGS
class Solution {
public:
    bool func(vector<int>& nums, int idx, int take, int not_take){
        if(idx == nums.size()){
            if(take != not_take)return false;
            return true;
        }
        // not_take
        bool a = func(nums, idx+1, take, not_take + nums[idx]);

        // take
        bool b = func(nums, idx+1, take + nums[idx], not_take);

        return a|b;
    }

    bool canPartition(vector<int>& nums) {
        return func(nums, 0, 0, 0);
    }
};

// Convertable to the above q1 as follows, condition: subset1_sum == subset2_sum => subset1_sum + subset2_sum == total and each of the = total/2 => need to find a subset with part sum = total/2
bool canPartition(vector<int>& nums) {
    int n = nums.size(), S = 0;
    if(n<=1)return false;
    for(int i = 0 ; i < nums.size(); i++) S+=nums[i];
    if(S % 2 != 0)return false;
    S = S/2;
    sort(nums.begin(),nums.end());
    if(nums[0]>S)return false;
    vector<vector<bool>>dp(n, vector<bool>(S+1, 0));
    for(int i = 0; i < n; i++)dp[i][0] = 0;
    dp[0][nums[0]] = true;
    for(int i = 1; i < n; i++){
        for(int t = 1; t <= S; t++){
            bool not_take = dp[i-1][t];
            bool take = false;
            if(nums[i] <= t)take = dp[i-1][t-nums[i]];
            dp[i][t] = take | not_take;
        }
    }
    return dp[n-1][S];
}

// Q7 - Coin Change
int change(int amount, vector<int>& coins) {
    vector<int> dp(amount + 1, 0);
    dp[0] = 1; // There's one way to make the amount 0 (using no coins)

    for (int coin : coins) {
        for (int i = coin; i <= amount; i++) {
            dp[i] += dp[i - coin];
        }
    }

    return dp[amount];
}