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
// Q1 - subsequence sum DP-14
// Does a subseq with sum exist in arr

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
        for(int i = 0; i < arr.size(); i++)dp[i][0] = true; //prev[0] - 1st col in 2d
        dp[0][arr[0]] = true;                               //prev[i] - 1st row in 2d - except 2 col 0 and arr[0] all are false
        for(int i = 1; i < arr.size(); i++){
            for(int t = 1; t <= sum; t++){
                bool not_take = dp[i-1][t];                 //prev[t]
                bool take = false;
                if(arr[i] <= t)take = dp[i-1][t-arr[i]];    //prev[t-arr[i]]
                dp[i][t] = take | not_take;                 //cur[t]
            }                                               //prev = cur
        }
        return dp[arr.size()-1][sum];                       //cur[sum]
    }

// 1D DP - Space Optimization
// since the above 2d tabulation has dp(i) <= dp(i-1) only and not i(or any other prev vales), we can use only 2 1d arr prev and cur.
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

// Q2 - Partition equal subset sum -DP- 15
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

    //Q1
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


// Q4 - Count Subsets with sum K
// if the arr contains only +ve nums - simple 

// if the arr contains non-neg int
// 2D DP Memoization
class Solution{
	public:
	int mod = 1e9 + 7;
    int func(vector<int> nums, int idx, int target, vector<vector<int>>& dp) {
        if (idx < 0) return 0;
    
        if (dp[idx][target] != -1) return dp[idx][target];
    
        int notTake = func(nums, idx - 1, target, dp);
        
        int take = 0;
        if (nums[idx] <= target) take = func(nums, idx - 1, target - nums[idx], dp);
    
        return dp[idx][target] = (take + notTake)%mod;
    }
    
    int perfectSum(int arr[], int n, int target) {
        vector<int> nums(arr, arr + n);
        sort(nums.begin(), nums.end());
        vector<vector<int>> dp(n, vector<int>(target + 1, -1));
    
        for (int i = 0; i < n; i++) dp[i][0] = 1; 
        if (nums[0] <= target) dp[0][nums[0]] = 1;
    
        int ans = func(nums, n - 1, target, dp);
         
        // the above code is enough if only for +ve, if a zero is present it doubles the subsets
        int zero = 0;
        for(int i = 0; i < n; i++){
            if(nums[i] > 0)break;
            zero++;
        }
        zero = (1 << zero);
        return (ans*zero)%mod;
    }
};

// striver
// 2D Memorization
class Solution{
	public:
    int mod = 1e9 + 7;
	int func(int idx, int sum, vector<int> &nums, vector<vector<int>>& dp){
	    if(idx == 0){
	        if(sum == 0 && nums[0] == 0)return 2;           // dont do if(sum == nums[i]) - thinking it will save time, its incorrect. can have ele 1,1,2 
	        if(sum == 0 || sum == nums[0])return 1;
	        return 0;
	    }
	    
	    if(dp[idx][sum] != -1) return dp[idx][sum];
	    
	    int notTake = func(idx-1, sum, nums, dp);
	    int take = 0;
	    if(nums[idx] <= sum)take = func(idx-1, sum-nums[idx], nums, dp);
	    
	    return dp[idx][sum] = (take+notTake)%mod;
	}
	
	int perfectSum(int arr[], int n, int sum){
	    vector<vector<int>> dp(n + 1, vector<int>(sum+1, -1));
	    vector<int> nums(arr, arr + n);
        return func(n-1, sum, nums, dp);
	}
};

// 2D Tabulation
class Solution{
int perfectSum(vector<int>& arr, int n, int sum) {
        vector<vector<int>> dp(n, vector<int>(sum + 1, 0));
    
        if (arr[0] == 0) dp[0][0] = 2; 
        else dp[0][0] = 1; 
    
        if (arr[0] != 0 && arr[0] <= sum) dp[0][arr[0]] = 1; 
            
        for (int i = 1; i < n; i++) {
            for (int k = 0; k <= sum; k++) {
                int notTake = dp[i - 1][k];
                int take = 0;
                if (arr[i] <= k) take = dp[i - 1][k - arr[i]];
                dp[i][k] = take + notTake;
            }
        }
    
        return dp[n - 1][sum];
    }
};

// Q5- Count Partitions with diff k - DP-18
// 2D memoization (TLE)
class Solution{
public:
    int func(vector<int> nums, int idx, int target, vector<vector<int>>& dp) {
        if(idx == 0){
	        if(target == 0 && nums[0] == 0)return 2;           // dont do if(sum == nums[i]) - thinking it will save time, its incorrect. can have ele 1,1,2 
	        if(target == 0 || target == nums[0])return 1;
	        return 0;
	    }
    
        if (dp[idx][target] != -1) return dp[idx][target];
        
        int notTake = func(nums, idx - 1, target, dp);
        
        int take = 0;
        if (nums[idx] <= target) take = func(nums, idx - 1, target - nums[idx], dp);

        return dp[idx][target] = (take + notTake);
    }
    
    int countPartitions(int n, int d, vector<int>& arr) {
        int total = 0;
        for(int i = 0; i < n; i++)total += arr[i]; 
        
        if(d%2 != total%2)return 0;
        int target = (d+total)/2;
        vector<vector<int>> dp(n, vector<int>(target+1, -1));
        
        return func(arr, n-1, target, dp);
    }
};

//from Q4 - striver  
class Solution{
  public:
    int mod = 1e9 + 7;
	int func(int idx, int sum, vector<int> &nums, vector<vector<int>>& dp){
	    if(idx == 0){
	        if(sum == 0 && nums[0] == 0)return 2;           // dont do if(sum == nums[i]) - thinking it will save time, its incorrect. can have ele 1,1,2 
	        if(sum == 0 || sum == nums[0])return 1;
	        return 0;
	    }
	    
	    if(dp[idx][sum] != -1) return dp[idx][sum];
	    
	    int notTake = func(idx-1, sum, nums, dp);
	    int take = 0;
	    if(nums[idx] <= sum)take = func(idx-1, sum-nums[idx], nums, dp);
	    
	    return dp[idx][sum] = (take+notTake)%mod;
	}
	
	int perfectSum(vector<int>& arr, int n, int sum){
	    vector<vector<int>> dp(n + 1, vector<int>(sum+1, -1));
        return func(n-1, sum, arr, dp);
	}

    int countPartitions(int n, int d, vector<int>& arr) {
        int total = 0;
        for(auto &it: arr)total += it;
        if(total -d < 0 || (total - d) % 2 ) return false;
        return perfectSum(arr, arr.size(), (total-d)/2);
    }
};

// Tabulation - didnot work
public:
int perfectSum(vector<int>& arr, int n, int sum) {
    vector<vector<int>> dp(n, vector<int>(sum + 1, 0));

    if(arr[0] == 0) dp[0][0] = 2;
    else dp[0][0] = 1; 
    
    if(arr[0] != 0 && arr[0] <= sum)dp[0][arr[0]] = 1;

    for (int i = 1; i < n; i++) {
        for (int k = 0; k <= sum; k++) {
            int notTake = dp[i - 1][k];
            int take = 0;
            if (arr[i - 1] <= k) take = dp[i - 1][k - arr[i]];
            dp[i][k] = take + notTake;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int k = 0; k <= sum; k++) {
            cout<<dp[i][k]<<" ";
        }
        cout<<endl;
    }

    return dp[n-1][sum];
}

int countPartitions(int n, int d, vector<int>& arr) {
    int total = accumulate(arr.begin(), arr.end(), 0);
    if (total < d || (total - d) % 2 != 0) return 0;

    return perfectSum(arr, n, (total - d) / 2);
}


// Q6 - 0-1 Knapsack - DP-19
// fractional knapsack - greedy - sort by val/wt and take 

// 2D Memoization
// dp[idx][w] = the max val of items of wt w that can be from val[0...idx] (w < W)
// time :O(2^n), space:O(nW+n)
class Solution{
    public:
    int func(int wt[], int val[], int n, int idx, int w, vector<vector<int>>& dp){
        if(idx == 0) return dp[0][w] = (wt[0] <= w)? val[0]:0;
        if(w == 0)return 0;
        if (dp[idx][w] != -1) return dp[idx][w];
        
        int notTake = func(wt, val, n, idx-1, w, dp);
        
        int take = 0;
        if(wt[idx] <= w) {
            take = func(wt, val, n, idx-1, w-wt[idx], dp);
            take += val[idx];
        }
        return dp[idx][w] = max(take, notTake);
    }

    int knapSack(int W, int wt[], int val[], int n) { 
        vector<vector<int>>dp(n, vector<int> (W+1, -1));
        return func(wt, val, n, n-1, W, dp);;
    }
};

// 2D tabulation
// time: O(n*W), space:O(n*W)
class Solution {
    public:
    int knapSack(int W, int wt[], int val[], int n) { 
        vector<vector<int>>dp(n, vector<int> (W+1, 0));
        for(int i = 0; i <= W; i++) dp[0][i] = (wt[0] <= i)? val[0]:0;
        for(int i = 0; i < n; i++) dp[i][0] = 0;
        
        for(int idx = 1; idx < n; idx++){
            for(int w = 1; w <= W; w++){
                int notTake = dp[idx-1][w];
                int take = 0;
                if(wt[idx] <= w) {
                    take = dp[idx-1][w-wt[idx]];
                    take += val[idx];
                }
                dp[idx][w] = max(take, notTake);
            }
        }
        return  dp[n-1][W];
    }
};

// Space Optimization - 2 row
// anytime dp[i][j] depends on dp[i-1][j]
// use prev cur dp method

class Solution {
    public:
    int knapSack(int W, int wt[], int val[], int n) { 
        vector<int> prev(W+1, 0), cur(W+1, 0);

        for(int w = 0; w <= W; w++) prev[w] = (wt[0] <= w)? val[0]:0;

        for(int idx = 1; idx < n; idx++){
            for(int w = 1; w <= W; w++){
                int notTake = prev[w];
                int take = 0;
                if(wt[idx] <= w) {
                    take = prev[w-wt[idx]];
                    take += val[idx];
                }
                cur[w] = max(take, notTake);
            }
            prev =cur;
        }
        return  prev[W];
    }
};

// Space Optimisation - 1 row
// cur[w] depends on prev[w] and prev[w-wt[i]] also cur[w] doesnt depend on any of the other valus in that arr(row/cur)
// using the same row 1.e prev instaed of using cur
// for cur[w] we req prev[w] and prev[w-wt[i]] so we can replace prev[w] with cur[w] and similarly prev[w-1] with cur[w-1] instead of using the the cur array
// but we can do it from left to right i.e fill prev[1] with cur[i] etc bcz in that case if cur[2] depends on prev[1] it is not more there, it has been override
class Solution {
    public:
    int knapSack(int W, int wt[], int val[], int n) { 
        vector<int> prev(W+1, 0);

        for(int w = 0; w <= W; w++) prev[w] = (wt[0] <= w)? val[0]:0;

        for(int idx = 1; idx < n; idx++){
            for(int w = W; w >= 0; w--){        //------------important change
                int notTake = prev[w];
                int take = 0;
                if(wt[idx] <= w) {
                    take = prev[w-wt[idx]];
                    take += val[idx];
                }
                prev[w] = max(take, notTake);
            }
            // prev = cur                       //------------important change
        }
        return  prev[W];
    }
};


// Q7 - minimum coins DP-20
// the min num of coins req to pay amount
// 1D Memoization
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        int n = coins.size();
        
        vector<int> dp(amount + 1, INT_MAX); 
        dp[0] = 0; 
        
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < n; j++) {
                if (coins[j] <= i && dp[i - coins[j]] != INT_MAX) {
                    dp[i] = min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }
        
        return dp[amount] == INT_MAX ? -1 : dp[amount];
    }
};

// Q8 - Target Sum - DP-21
// all the ele in nums can either be add or sub to get target
// 2D - Memoization
// time:O(2^n), space:O(n*(2*target+1))
class Solution {
public:
    int func(vector<int>& nums, int target, int idx, vector<vector<int>>& dp){
        int offset = dp[0].size()/2;
        if(offset+target >= dp[0].size() || offset+target < 0) return 0;
        if(idx == 0 && nums[0] == 0 && target == 0)return 2;                //------------- edge case 
        if(idx == 0) return dp[0][offset+target] = (target == nums[0] || target == 0-nums[0])? 1:0;

        if(dp[idx][offset+target] != -1) return dp[idx][offset+target];

        int plus = func(nums, target-nums[idx],idx-1, dp);
        int minus = func(nums, target+nums[idx], idx-1, dp);
        return dp[idx][offset+target] = plus + minus;
    }

    int findTargetSumWays(vector<int>& nums, int target) {
        int n = nums.size();
        int sum = 0;
        for(int i = 0 ; i < n; i++){
            sum += nums[i];
        }

        vector<vector<int>>dp(n, vector<int>(2*sum+1, -1));
        return func(nums, target, n-1, dp);
    }
};

// 2D tabulation
// 
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        int n = nums.size();
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        
        if (abs(target) > sum) return 0;  // If the target is out of possible sum range, return 0.
        
        vector<vector<int>> dp(n, vector<int>(2 * sum + 1, 0));
        
        // Initialize base cases
        dp[0][sum + nums[0]] += 1;  // Case for adding nums[0]
        dp[0][sum - nums[0]] += 1;  // Case for subtracting nums[0]
        
        
        for (int i = 1; i < n; ++i) {
            for (int j = -sum; j <= sum; ++j) {
                if (sum + j - nums[i] >= 0 && sum + j - nums[i] < 2 * sum + 1) {
                    dp[i][sum + j] += dp[i - 1][sum + j - nums[i]];
                }
                if (sum + j + nums[i] >= 0 && sum + j + nums[i] < 2 * sum + 1) {
                    dp[i][sum + j] += dp[i - 1][sum + j + nums[i]];
                }
            }
        }
        
        return dp[n - 1][sum + target];
    }
};
// or
// can also have 
// if (dp[i - 1][sum + j] > 0) {
//     dp[i][sum + j + nums[i]] += dp[i - 1][sum + j];
//     dp[i][sum + j - nums[i]] += dp[i - 1][sum + j];
// }

// Space Optimization 2 rows
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        int n = nums.size();
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        
        if (abs(target) > sum) return 0;  // If the target is out of possible sum range, return 0.
        
        vector<int> prev(2 * sum + 1, 0), cur(2 * sum + 1, 0);
        
        // Initialize base cases
        prev[sum + nums[0]] += 1;  // Case for adding nums[0]
        prev[sum - nums[0]] += 1;  // Case for subtracting nums[0]
        if(n == 1)return prev[sum + nums[0]];

        for (int i = 1; i < n; ++i) {
            fill(cur.begin(), cur.end(), 0);  // Reset cur array
            for (int j = -sum; j <= sum; ++j) {
                if (sum + j - nums[i] >= 0 && sum + j - nums[i] < 2 * sum + 1) {
                    cur[sum + j] += prev[sum + j - nums[i]];
                }
                if (sum + j + nums[i] >= 0 && sum + j + nums[i] < 2 * sum + 1) {
                    cur[sum + j] += prev[sum + j + nums[i]];
                }
            }
            prev = cur;
        }
        return cur[sum + target];
    }
};


// Q9 - Coin Change - DP -22
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

// Q10- Unbounded knapsack - DP - 23
// same as 0-1 knapsack except, in case of take, the idx remains at the same idx doesnt dec - because even after taking an item, he can take it again
// the above cond. applies for all infinite supply/unbounded dp problems.
// 2D Memoization
// time:O(2^n), space: O(W)
class Solution{
public:
    int func(int wt[], int val[], int n, int idx, int w, vector<vector<int>>& dp){
        if(idx == 0) {
            return (w/wt[0])*val[0];
        }
        if(w == 0)return 0;
        if (dp[idx][w] != -1) return dp[idx][w];
        
        int notTake = func(wt, val, n, idx-1, w, dp);
        
        int take = 0;
        if(wt[idx] <= w) {
            take = func(wt, val, n, idx, w-wt[idx], dp);
            take += val[idx];
        }
        return dp[idx][w] = max(take, notTake);
    }
    
    int knapSack(int n, int W, int val[], int wt[]){
        vector<vector<int>>dp(n, vector<int> (W+1, -1));
        return func(wt, val, n, n-1, W, dp);
    }
};

// Tabulation
class Solution{
public:
    
    int knapSack(int n, int W, int val[], int wt[]){
        vector<vector<int>>dp(n, vector<int> (W+1, 0));
        
        for(int w = 0; w <= W; w++) dp[0][w] = (w/wt[0])*val[0];
        
        for(int idx = 1; idx < n; idx++){
            for(int w = 0; w <= W; w++){
                int notTake = dp[idx-1][w];
                int take = 0;
                if(wt[idx] <= w) take = dp[idx][w-wt[idx]] + val[idx];
                dp[idx][w] = max(take, notTake);
            }
        }
        
        return dp[n-1][W];
    }
};

// Space Optimisation to 2 1D arrays - when dp(idx) <= dp(idx-1) and/or d(idx)
class Solution{
public:
    int knapSack(int n, int W, int val[], int wt[]){
        vector<int> prev(W+1, 0), cur(W+1,0);
        
        for(int w = 0; w <= W; w++) prev[w] = (w/wt[0])*val[0];
        
        for(int idx = 1; idx < n; idx++){
            for(int w = 0; w <= W; w++){
                int notTake = prev[w];
                int take = 0;
                if(wt[idx] <= w) take = cur[w-wt[idx]] + val[idx];
                cur[w] = max(take, notTake);
            }
            prev = cur;
        }
        
        return prev[W];
    }
};

// space optimization to 1D - just update the same cur[w]  - kinda like the prev[w] gets rewritten by the max of prev[w] and prev[w-wt[i]] 
class Solution{
public:
    
    int knapSack(int n, int W, int val[], int wt[]){
        vector<int> cur(W+1,0);
        
        for(int w = 0; w <= W; w++) cur[w] = (w/wt[0])*val[0];
        
        for(int idx = 1; idx < n; idx++){
            for(int w = 0; w <= W; w++){
                int notTake = cur[w];
                int take = 0;
                if(wt[idx] <= w) take = cur[w-wt[idx]] + val[idx];
                cur[w] = max(take, notTake);
            }
        }
        
        return cur[W];
    }
};