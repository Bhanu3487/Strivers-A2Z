#include <iostream>
using namespace std;
#include <vector>
#include <bits/stdc++.h>

// Q2. Array(easy) 
// [ second largest element, second smalled element]
// brute force: 2 passes
vector<int> getSecondOrderElements(int n, vector<int> a) {
    int max = INT_MIN, max_2 = INT_MIN, min = INT_MAX, min_2 = INT_MAX;
        for(int i = 0; i < n; i++){
        if (a[i] > max) {
            max = a[i];
        }
        if (a[i] < min) {
            min = a[i];
        }
        }

        for(int i = 0; i < n; i++){
        if (a[i] > max_2 && a[i] != max) {
            max_2 = a[i];
        }
        if (a[i] < min_2 && a[i] != min) {
            min_2 = a[i];
        }
    }

    vector<int> vect{ max_2, min_2 };
    return vect;
}
// better : 1 pass
	int print2largest(int arr[], int n) {
	    int lar = -1, sec = -1;
	    for(int i = 0; i < n; i++){
	        if(arr[i] > lar){
	            if(lar > sec)sec = lar;
	            lar = arr[i];
	        }
	        else if(arr[i] > sec && arr[i] < lar){
	            sec = arr[i];
	        }
	    }
	    return sec;
    }

// Q3 
// sorted array shifted by x
// find min and check if arr[min+idx] <= arr[min+idx+1]
// time : O(n), space : O(1)
class SolutionQ3 {
public:
    bool check(vector<int>& nums) {
        int x = 0, min = 0, n = nums.size();
        for(int i = 1; i < n; i++){     // find min
            if (nums[i] <= nums[min]){
                min = i;
            }
        }
        while (x < n - 1){      
            if(nums[(min + x) % n] <= nums[(min + x + 1) % n]){
                x++;
            }
            else return false;
        }
        return true;
    }
};

//Q4 Remove duplicates from sorted array
// search for last repeated element and store arr[unique] = arr[repeated] 
// arr[0 .. unique] is ans, time : O(n), space: O(1)
class SolutionQ4 {
public:
    int removeDuplicates(vector<int>& nums) {
        int repeated = 1, unique = 1;
        while(repeated < nums.size()){
            if(nums[repeated-1] != nums[repeated]){
                nums[unique] = nums[repeated];
                unique++; 
            }
            repeated++;
        }
        return unique;
    }
};

// Q5 Left Rotate an array by one place - easy

// Q6 Left rotate an array by D places
// method 1- run q5 k times, time: O(nk), space: O(1)
// method 2- store 1st k ele in a new arr, shift n-k ele in arr to left and paste the first k ele.
// time : O(n), space: O(k)
vector<int> rotateArray(vector<int>arr, int k) {
    // Write your code here.
    int n = arr.size();
    int arr2[k];
    for(int i = 0; i < k; i++){
        arr2[i] = arr[i];
    }
    for(int i = 0; i < n; i++){
        if(i < n-k) arr[i] = arr[i+k];
        else arr[i] = arr2[i - n + k];
    }
    return arr;
}

// Q7 Move zeros to the end
// 2 pointer method, start and end mark the start and end of zeroes till idx+1, exchange a[start] and a[end+1]
// time : O(n) space : O(1)
class Solution {
public:
    void moveZeroes(vector<int>& a) {
        int start = -1, end = -1, idx = 0, n = a.size();
        while(end < n && idx < n){
            if(a[idx] == 0 && start == -1){
                start = idx;
            }
            if(a[idx] == 0 && ((idx+1) < n) &&a[idx+1] != 0){
                end = idx;
            }
            if(end != -1 && a[idx] != 0){
                a[start] = a[idx];
                a[idx] = 0;
                start++;
                end = idx;
            }
            idx++;
        }
    }
};

// better method
// j contains the idx of 1st zero, we search for non zero ele and exchange with a[j]. 
// update j++; bcz if originally j+1 has a zero fine, else due to swap we get a zero at j+1
// time: O(n), space: O(1)
vector<int> moveZeroes(int n, vector<int> a){
    int j = -1;
    for(int i = 0; i < n; i++){
        if(a[i] == 0){
            j = i; break;
        }

        if(j == -1) return a;   //arr has no zero

        for(int i = j+1; i < n; i++){
            if(a[i] != 0){
                swap(a[i], a[j]);
                j++;
            }
        }
    }
    return a;
}

// Q8 Union of 2 sorted arrays with duplicates
// create merged array (as in merge sort)
// 2 pointer we push the smaller of a[i] and b[j] such that result doesnt contain them.
// time: O(n+m), space: O(n+m)
vector < int > sortedArray(vector < int > &a, vector < int > &b) {
    vector<int> result;
    int i = 0, j = 0, k = 0;
    while(i < a.size() && j < b.size()){
      if (a[i] < b[j]) {
        if(a[i] != k){
          result.push_back(a[i]);
          k = a[i];
        }
        i++;
      } 
      if (a[i] > b[j]){
        if (b[j] != k) {
          result.push_back(b[j]);
          k = b[j];
        }
        j++;
      }
      if (a[i] == b[j]){
        if (a[i] != k) {
          result.push_back(b[j]);
          k = b[j];
        }
        j++;
        i++;
      }
    }

    while(j < b.size()){
      if(b[j] != b[j-1]) result.push_back(b[j]);
      j++;
    }
    while(i < a.size()){
      if(a[i] != a[i-1]) result.push_back(a[i]);
      i++;
    }
    return result;
}

// another method
// convert arr a and b into sets - removes duplicates and then merge them
// time: O(n+m)log(n+m) , space : O(n+m)


// Feb 5
// Q10 
// missing number in an array containing [0,n] missing 1 from that range
// sub all ele from sum
//alt: xor ans with ele and arr[ele] or binary search

class SolutionQ10 {
public:
    int missingNumber(vector<int>& nums) {
        int n = nums.size();
        int sum = (n*(n+1))/2;
        for(int i = 0; i < n; i++){
            sum -= nums[i];
        }
        return sum;
    }
};

//Q11
//Maximum Consecutive Ones
//time: O(n); space: O(1)
class SolutionQ11 {
public:
    int findMaxConsecutiveOnes(vector<int>& nums) {
        int ans = 0, cur = 0;
        for(int i = 0; i < nums.size(); i++){
            if(ans < cur) ans = cur;
            if(nums[i] == 1)cur++;
            else cur = 0;
        }
        if(ans < cur) ans = cur;
        return ans;
    }
};

//Q12
// single number: find the only num with freq 1 rem have freq 2
// xor all ele
// alt: storing map<ele,freq> time: O(n); space: O(n)
//      sorting , time : O(nlogn) ; space : O(1)
//      sum of ele: ans = sum(unique_ele*2) - sum(all_ele); time: O(n), space : O(n) to store unique ele
class SolutionQ12 {
public:
    int singleNumber(vector<int>& nums) {
        int ans = 0;
        for(int i = 0; i < nums.size(); i++){
            ans = ans ^ nums[i];
        }
        return ans;
    }
};

//Q13 optimal solution by take u frwd

int longestSubarrayWithSumK(vector<int> a, long long k) {
    int left = 0, right = 0;
    long long sum = a[0];
    int max_len = 0;
    int n = a.size();
    while(right < n){
        while(left <= right && sum > k){
            sum -= a[left];
            left++;
        }
        if(sum == k) max_len = max(max_len, right-left+1);
        right++;
        if(right < n) sum += a[right];
    }
    return max_len;
}

//better solution by tuf for +ves, and optimal for all integers
// in an ordered map we store sum[0...i] and i, using this we can find sum[i...j] = sum[0...j] - sum[0...i]; 
//didnt run in cninjas prolly bcz map related error; note: ordered map O(nlogn) sort?? and unordered O(n*1)best , O(n*n)worst
int longestSubarrayWithSumK(vector<int> a, long long k) {
    map<long long, int> preSumMap;
    long long sum = 0;
    int maxLen = 0;
    for(int i = 0; i < a.size(); i++){
        sum+=a[i];
        if(sum == k) maxLen = max(maxLen, i+1);
        long long rem = k - sum;
        if(preSumMap.find(rem) != preSumMap.end()){
            int len = i - preSumMap[rem];
            maxLen = max(maxLen, len);
        }
        if(preSumMap.find(sum) == preSumMap.end()) {preSumMap[sum] = i;}
    }
    return maxLen;
}

//works only if elements are +ve:
//time: O(n) space = O(1)
int longestSubarrayWithSumK(vector<int> a, long long k) {
    int n = a.size();
    int sum = 0;
    int len = 0;
    int ans = 0;
    for(int i = 0; i < n; i++){
        if(a[i] + sum == k && ans <= len){
            sum = k;
            ans = ++len;
        }
        else if(a[i] + sum < k){
            len++;
            sum += a[i];
        }
        else {
            sum = a[i];
            len = 1;
            if( ans <= len) ans = len;
        }
    }
    return ans;
}

//-------------------------------------------MEDIUM-----------------------------------------------------

//Q1 2sumtarget
//brute force: i (from i to n-1) j(from i+1 to n) if a[i] + a[j] == target,
//time: O(n^2), space: O(1)

//better sol'n using hashmaps
//map {num,idx} sorted by num, 2 ptrs from left and right, comapre (a[left] + a[right]) with target
//time: O(nlogn) sort (if unorderd map O(n^2) worst case), space: O(n)
class SolutionQ1_1 {
public:
    static bool compareFunc(pair<int,int>& a, pair<int,int>& b){
        return a.first < b.first;
    }

    vector<int> twoSum(vector<int>& nums, int target) {
        vector<pair<int, int>> idx(nums.size());
        for(int i = 0; i < nums.size(); i++){
            idx[i].first = nums[i];
            idx[i].second = i;
        }

        sort(idx.begin(), idx.end(), compareFunc);

        int left = 0, right = idx.size() - 1; // Initialize right to the last index
        
        while(left < right){
            if(idx[left].first + idx[right].first == target) 
                return {idx[left].second, idx[right].second};
            else if (idx[left].first + idx[right].first < target) 
                left++;
            else 
                right--;
        }
        return {};
    }
};

//without sorting, we can create a map with (num,idx) and if target-num exists in  the map we return idxs
//time: O(n); space: O(n)
class SolutionQ1 {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> numMap;
        int n = nums.size();

        for (int i = 0; i < n; i++) {
            int complement = target - nums[i];
            if (numMap.count(complement)) {
                return {numMap[complement], i};
            }
            numMap[nums[i]] = i;
        }

        return {}; // No solution found
    }
};

//Q2 Sort 3 colors
//brute sort, time: O(nlogn) space:O(n) for merge sort

//medium (by striver, mine):count all diff colors and store in var, replace values in arr
//time: O(n), space: O(1)
class SolutionQ2_1 {
public:
    void sortColors(vector<int>& nums) {
        int zeroes = 0, ones = 0, twos = 0;
        for(int i = 0; i < nums.size(); i++){
            if(nums[i] == 0) zeroes++;
            else if(nums[i] == 1) ones++;
            else twos++;
        }
        int idx = 0;
        while(idx < nums.size()){
            if(idx < zeroes) nums[idx] = 0;
            else if(idx < zeroes + ones) nums[idx] = 1;
            else nums[idx] = 2;
            idx++;
        }
    }
};

//optimal soln striver
//use 3 ptrs 
//time: O(n), space: O(1)
class SolutionQ2 {
public:
    void sortColors(vector<int>& nums) {
        int n = nums.size();
        int low = 0, mid = 0, high = n-1;
        while(low <= mid && mid <= high){
            if(nums[mid] == 0){
                swap(nums[low], nums[mid]);
                low++, mid++;
            }
            else if(nums[mid] == 1) mid++;
            else {
                swap(nums[mid], nums[high]);
                high--;
            }
        }
    }
};

// Q3
// find the majority element (freq>n/2)
// brute force find freq of each element and return when freq == n/2, time: O(n^2) and space = O(1)
// better : using hashmaps: map[arr[i]]++, time = O(n) and space: O(n)
// optimal solution : Moore voting algorithm; time: O(n) nd space : O(1)
// c is the ans for [i...j], here v becomes 0 for idx i and j
// initialise c to be the 1st element, and 1 to v each time c is seen
// if i != c we sub 1 from v. if [1...i] has 3 'a' and 3 other diff elements
// then c gets incremented by 1, 3 times and decremented by 3 times and again c gets initialsed as the arr[i+1]
class SolutionQ3 {
public:
    int majorityElement(vector<int>& nums) {
        int c = 0, v = 0;
        for (auto i:nums){
            if (v == 0) c = i;
            if (c == i) v++;
            else v--;
            cout << i << c << v << endl;
        }
        return c;
    }
};

// Q4
// find the subset with maximum sum
// brute for i <- 0 to n-1 and j <- 1 to n+1
// find the sum of all [i...j-1] time : O(n^2) and space = O(1)
// better: Kadane's algorithm
// 2 variables= max with the max sum of subset till idx and sum is the sum till idx from non neg ele
//time: O(n), space : O(1)

class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int max = INT_MIN;
        int sum = 0;
        for(auto i:nums){
            sum += i;
            if(sum > max) max = sum;
            if(sum<0)sum = 0;
        }
        return max;
    }
};

// Optimal??
// Divide and conquer method

// Q6 BUY AND SELL STOCKS - single day
// buy on ith day and sell on i+k th day - what is max profit
// DP problem, dp[i][i+k] = price[i+k] - price[i] , time : O(n^2), space = O(n^2)
// idea: at idx i, we track the min price till ith day and we update max profit if it is less than price[i] - min
// time: O(n) space : O(1)
#include <vector>
#include <algorithm>

int bestTimeToBuyAndSellStock(std::vector<int>& prices) {
    int n = prices.size();
    if (n < 2) return 0; 
    
    int min_price = prices[0]; 
    int max_profit = 0; 

    for (int i = 1; i < n; ++i) {
        max_profit = max(max_profit, prices[i] - min_price); 
        min_price = min(min_price, prices[i]);
    }
    
    return max_profit;
}

//Q7 Rearrange the array in alternating positive and negative items
// arr has equal pos and neg, return arr st it has alternate pos and neg, starting with pos. 
// brute, store pos and neg elements in different arrs and replace even and odd idx in nums with pos and neg ele in order
// time: O(n) (2 passes) space: O(n) (2 arr, pos and neg of size n/2)
// optimal, create an arr ans and store pos ele in nums at even idx in ans, opp for neg
// time: O(n) (1 pass) space: O(n)
// asked for O(1) here space can be taken to be O(1) bcz output is vector<int> so no extraspace created
class Solution {
public:
    vector<int> rearrangeArray(vector<int>& nums) {
        vector<int> pos, neg; 
        for(int i = 0; i < nums.size(); i++){
            if(nums[i] > 0) pos.push_back(nums[i]);
            else neg.push_back(nums[i]);
        }
        int p = 0, n = 0;
        for(int i = 0; i < nums.size(); i++){
            if(i%2==0){
                nums[i] = pos[p];
                p++;
            }
            else {
                cout << neg[n];
                nums[i] = neg[n];
                n++;
            }
        }
        return nums;
    }
};


// Q8 Next greater Permutation
// from end, find idx which is greater than its next ele and mark that idx, if none(last permutation) return ascending arr
// now from right find the first element which is greater than A[idx] (1st bcz idx+1 to n-1 is in decreasing order so 1st is smallest num greater than A[idx])
// now swap the above ele and idx, and reverse the part from idx+1 to end.
// time: O(n), space: O(1) 

class Solution {
public:
    void nextPermutation(vector<int>& A) {
        int ind = -1;
        int n = A.size();
        for(int i = n-2; i >=0 ; i--){
            if(A[i] < A[i+1]){
                ind = i;
                break;
            }
        }
        if(ind == -1) {
            reverse(A.begin(), A.end());
            return;
        }
        for(int i = n-1; i > ind; i--){
            if(A[i] > A[ind]){
                swap(A[i], A[ind]);
                break;
            }
        }

        reverse(A.begin()+ind+1, A.end());
    }
};

//Q9- easy

// Q10 Longest Consecutive sequence
// brute sol'n for x in arr check for x+1 if exists a while loop to search for x+2...
// time : O(n**2), space : O(1)
// better sol'n
// sort the array, in a for loop till any arr[i+1] - arr[i] != 1 or reached n-1, continue increasing the count. also remove repeated ele.
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        int n = nums.size();
        if (n==1) return 1;
        sort(nums.begin(), nums.end());
        int ans = 0, diff = 0, start = 0, rep = 0;
        for(int i = 0; i < n; i++){
            if(i != n-1 && nums[i+1] == nums[i]){
                rep++;
            }
            else if(i == n-1 || nums[i+1] - nums[i] != 1){
                diff = i - start + 1 - rep;
                if(diff > ans) ans = diff;
                start = i+1;
                rep = 0;
            }
        }
        return ans;
    }
};
// optimal sol'n: using unordered set
// store all ele in a set 
// for an ele in set if its prev num exists in set continue, else start checking for next ele in while loop(as in brute force)
// searching for ele in set: best/avg case: O(1) , worst : O(n) 
// time, 1st for loop :O(n), 2nd for loop with while loop : O(n) (2 passes kind of)
// for numbers [100,4,101,2,102,3,1] : 2nd for loop enters only for 100 (while loop iters 3) and 1(iters 4)- total 7
// space :O(n) for set
class Solution {
public:
    int longestConsecutive(vector<int>& a) {
        int n = a.size();
        if (n<2) return n;
        int longest = 1;
        unordered_set<int> st;
        for(int i = 0; i < n; i++){
            st.insert(a[i]);
        }
        for(auto it:st){
            if(st.find(it - 1) == st.end()){
                int cnt = 1;
                int x = it;
                while(st.find(x+1) != st.end()){
                    x = x+1;
                    cnt = cnt + 1;
                }
                longest = max(longest, cnt);
            }
        }
        return longest;
    }
};

// Q11 Set Matrix Zeroes
// input 2d matrix, if m[i][j] == 0 then set the whole row and coln to zero
// brute force: have 2 unordered sets keep track of rows and col'n which should be made zeroes.
// time: O(n^2), space : O(n)
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int n = matrix.size(), m = matrix[0].size(); 
        unordered_set<int> rows,cols;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                if(matrix[i][j] == 0){
                    rows.insert(i);
                    cols.insert(j);
                }
            }
        }
        if(rows.empty()) return;

        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                if(rows.find(i) != rows.end()) matrix[i][j] = 0;
                else if(cols.find(j) != cols.end()) matrix[i][j] = 0;
            }
        }
    }
};
// optimal sol'n: 
// similar to the prev one but we use the 1st row(with an extra variable as m[0][0] is would repeat other wise) and 1st coln to keep track of zeroes
// and we check from right bottom corner to prevent overwriting incorrect zeroes because of col variable which is added
// time: O(n^2), space : O(1)
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int n = matrix.size(), m = matrix[0].size(); 
        int col_0 = 1;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                if(matrix[i][j] == 0){
                    if(j > 0) {matrix[i][0] = 0; matrix[0][j] = 0;}
                    else {matrix[i][0] = 0; col_0 = 0;}
                }
            }
        }

        for(int i = n-1; i >= 0; i--){
            for(int j = m-1; j >= 1; j--){
                if(matrix[i][0] == 0 || matrix[0][j] == 0){
                    matrix[i][j] = 0;
                }
            }
        }
        if(col_0 == 0){
            for(int i = 0; i < n; i++) matrix[i][0] = 0;
        }
    }
};

// Q13 Rotate matrix by 90
// like a rubiks cube right turn
// idea: elements at matrix[i][j] its image wrt x axis, y axis and origin (with the center of matric as origin) should make a clockwise rotation
// time: (n^2) space : O(1)
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        if(n%2 == 1){
            for(int i = 0; i < (n/2); i++){
                for(int j = i; j < n-1-i; j++){
                    int temp = matrix[i][j];
                    matrix[i][j] = matrix[n-1-j][i];
                    matrix[n-1-j][i] = matrix[n-1-i][n-1-j];
                    matrix[n-1-i][n-1-j] = matrix[j][n-1-i];
                    matrix[j][n-1-i] = temp;
                }
            }
        }
        else {
            for(int i = 0; i < (n/2)+1; i++){
                for(int j = i; j < n-1-i; j++){
                    int temp = matrix[i][j];
                    matrix[i][j] = matrix[n-1-j][i];
                    matrix[n-1-j][i] = matrix[n-1-i][n-1-j];
                    matrix[n-1-i][n-1-j] = matrix[j][n-1-i];
                    matrix[j][n-1-i] = temp;
                }
            }
        }
        
        
    }
};

//Q14 Spiral order of matrix
// just find the pattern for mxn matric
// time: O(mxn); space ; O(mxn)
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<int> ans;
        int i = 0, row = 0, col = 0;
        while(true){
            if(i > n-i-1) break;
            for(int col = i; col < n-i; col++){
                ans.push_back(matrix[i][col]);
            }
            if(i+1 > m-i-1) break;
            for(int row = i+1; row < m-i; row++){
                ans.push_back(matrix[row][n-1-i]);
            }
            if(n-2-i < i) break;
            for(int col = n-2-i; col > i-1; col--){
                ans.push_back(matrix[m-1-i][col]);
            }
            if(m-2-i < i+1) break;
            for(int row = m-2-i; row > i; row--){
                ans.push_back(matrix[row][i]);
            }
            i++;
        }
        return ans;
    }
};


//Q15 - sub array sum - using prefix sum
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int n = nums.size();
        int dp_table[n+1];
        dp_table[0] = 0;
        dp_table[1] = nums[0];
        for(int i = 2; i < n+1; i++){
            dp_table[i] = dp_table[i-1] + nums[i-1];
        }
        int count = 0;
        for(int j = n; j > -1; j--){
            for(int i = 0; i < j; i++){
                if(dp_table[j] - dp_table[i] == k) count++;
                cout << i << " " << j << " " << dp_table[j] - dp_table[i] << endl;
            }
        }
        return count;
    }
};


//------------------------------ HARD ---------------------------------------
// Q1: Pascals Triangle
// try on book- check for pattern- write as a lower right angled triangle than an  isoceles triangle
// time: O(n^2), space : O(n^2)
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> triangle;
        if(numRows<=0) return triangle;
        for(int i = 0; i < numRows; i++){
            vector<int> row;
            for(int j = 0; j < i+1; j++){
                if(j==0 || j==i) row.push_back(1);
                else row.push_back(triangle[i-1][j-1] + triangle[i-1][j]);
            }
            triangle.push_back(row);
        }
        return triangle;
    }
};

//Q2: majority element - freq >n/2
//used unordered map to store freq 
// better: time: O(n)- n*O(1) - unordered map(best case, O(n) for worst case), space: O(n)
class Solution {
public:
    vector<int> majorityElement(vector<int>& nums) {
        vector<int> ans;
        unordered_map<int, int> freq; 
        int n = nums.size(); 
        
        for(int i = 0; i < n; i++) {
            freq[nums[i]]++;
        }
        
        for (auto it = freq.begin(); it != freq.end(); it++) {
            if (it->second > n/3) {
                ans.push_back(it->first);
                if(ans.size() > 2) break;}
        }
        
        return ans;
    }
};

//optimal: Boyer-Moore Majority Vote - reduce space
// time: O(n), space: O(1)
// not necessary that e1 and e2 are answers so need to 2nd pass check

class Solution {
public:
    vector<int> majorityElement(vector<int>& nums) {
        int n = nums.size();
        vector<int> ans;
        int c1 = 0, e1 = 0;
        int c2 = 0, e2 = 0;
        for(int i = 0; i < n; i++){
            if(c1 == 0 && (c2 == 0 || e2 != nums[i])){
                c1++; e1 = nums[i];
            }
            else if(e1 == nums[i]){ 
                c1++;
            }
            else if(c2 == 0){
                c2++; e2 = nums[i];
            }
            else if(e2 == nums[i]){ 
                c2++;
            }
            else{c1--; c2--;}
        }
        c1 = c2 = 0;        //2nd pass check
        for (int i = 0; i < n; ++i) {
            if (nums[i] == e1) c1++;
            else if (nums[i] == e2) c2++;
        }
        if (c1 > n / 3) ans.push_back(e1);
        if (c2 > n / 3) ans.push_back(e2);
        return ans;
    }
};

// Q3 3sum - num[i]+num[j]+num[k] = 0 and i,j,k are unique
// 2 loops - fix i, then fix j(>i) st k lies between them
// this prevents repetition, set hashset to make search easier, set 
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = nums.size();
        set<vector<int>> st;
        
        for(int i = 0; i < n; i++){
            set<int> hashset;
            for(int j = i+1; j < n; j++){
                int third = -(nums[i] + nums[j]);
                if(hashset.find(third) != hashset.end()){
                    vector<int> temp = {nums[i], nums[j], third};
                    sort(temp.begin(), temp.end());
                    st.insert(temp);
                }
                hashset.insert(nums[j]);
            }
        }
        vector<vector<int>> ans(st.begin(), st.end());
        return ans;
    }
};


