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

// Q4 - sort a stack
//time:O(n^2), Auxilliary Space: O(N) recursive
class SortedStack{
public:
	stack<int> s;
	void sort();
};

void insert_at_bottom(stack<int> &s, int x){
    if(s.empty()){
        s.push(x);
    }
    else{
        int a = s.top();
        s.pop();
        if(x < a){
            insert_at_bottom(s, x);
            s.push(a);
        }   
        else{
            insert_at_bottom(s, a);
            s.push(x);
        }
    }
} 

void SortedStack :: sort(){
   if(!s.empty()){
        int x = s.top();
        s.pop();
        sort();
        insert_at_bottom(s, x);
    }
}

// Q5 - reverse a stack
// reverse is called each time to store each value and empty the st, once emptied, we insert the right nums to that point and recursively call it...
// time:O(n**2), auxillary space: O(1) - we use recursice stack space
void insert_at_bottom(stack<int> &St, int x){
        if(St.empty()){
            St.push(x);
        }
        else{
            int a = St.top();
            St.pop();
            insert_at_bottom(St, x);
            St.push(a);
        }
    }
    void Reverse(stack<int> &St){
        if(!St.empty()){
            int x = St.top();
            St.pop();
            Reverse(St);
            insert_at_bottom(St, x);
        }
    }

// Recursive Sequences -------------------
// Q1 - generate all binary strings of len k with no consecutive 1's
// n counts the size at each call

void generateAllStringsUtil(int K, char str[], int n){
    if (n == K){
        str[n] = '\0' ;
        cout << str << " ";
        return ;
    }

    if (str[n-1] == '1'){
        str[n] = '0';
        generateAllStringsUtil (K , str , n+1);
    }

    if (str[n-1] == '0'){
        str[n] = '0';
        generateAllStringsUtil(K, str, n+1);
        str[n] = '1';
        generateAllStringsUtil(K, str, n+1) ;
    }
}


void generateAllStrings(int K )
{
	if (K <= 0)return ;

	char str[K];
	str[0] = '0' ;
	generateAllStringsUtil ( K , str , 1 ) ;

	str[0] = '1' ;
	generateAllStringsUtil ( K , str , 1 );
}

// Q2 - Generate pranthesis
//The idea is intuitive. Use two integers to count the remaining left parenthesis (n) and the right parenthesis (m) to be added
// At each function call add a left parenthesis if n >0 and add a right parenthesis if m>0
// Append the result and terminate recursive calls when both m and n are zero.
// class Solution {
vector<string> generateParenthesis(int n) {
    vector<string> res;
    addingpar(res, "", n, 0);
    return res;
}
void addingpar(vector<string> &v, string str, int n, int m){
    if(n==0 && m==0) {
        v.push_back(str);
        return;
    }
    if(m > 0){ addingpar(v, str+")", n, m-1); }
    if(n > 0){ addingpar(v, str+"(", n-1, m+1); }
}

// Q3 - print all subsequences/power set of nums
class Solution {
public:
    vector<vector<int>> helper(vector<int> &v, int i){
        if(i >= v.size())return {{}};
        
        vector<vector<int>> partialans = helper(v, i+1);
        vector<vector<int>> ans;

        for(vector<int> x:partialans){
            ans.push_back(x);
        }
        for(vector<int> x:partialans){
            x.push_back(v[i]);
            ans.push_back(x);
        }
        return ans;
    }

//or
class Solution {
public:
    void helper(vector<int> &v, int i, vector<int> subset, vector<vector<int>> &ans){
        if(i == v.size()){
            ans.push_back(subset);
            return;
        }
        // not take
        helper(v, i+1, subset, ans);
        // take
        subset.push_back(v[i]);
        helper(v, i+1, subset, ans);
        return;
    }


    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> ans;
        vector<int> subset;
        helper(nums, 0, subset, ans);
        return ans;
    }
};


    vector<vector<int>> subsets(vector<int>& nums) {
        return helper(nums, 0);
    }
};

// Q7 Combination sum
// given a vec candidates with unique nums and target, what all combination of candidates(repeated) can be found to sum to target
// vec {2,3,5} t = 8: [[2,2,2,2], [2,3,3], [3,5]]
// have idx and target as parameters in helper func, if(target == 0)- we push the subset into ans; 
// at each idx, we call helper for idx <- idx to n-1, idx bcz it can be repeated. not from 0 to n-1 to prevent repeated subsets
// after backtracking we pop from the vector, to prevent the use of multiple vecs
// time: 
class Solution {
public:
    void helper(vector<int>& candidates, int target, vector<int>& subset, vector<vector<int>>& ans, int start) {
        if (target == 0) {
            ans.push_back(subset);
            return;
        }

        for (int i = start; i < candidates.size(); i++) {       //prevents duplicates in ans
            if (target >= candidates[i]) {
                subset.push_back(candidates[i]);
                helper(candidates, target - candidates[i], subset, ans, i); // not i + 1 because we can reuse same elements
                subset.pop_back();
            }
        }
    }

    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> ans;
        vector<int> subset;
        helper(candidates, target, subset, ans, 0);
        return ans;
    }
};

// Q8 Conbination sum 2
// given vec with int (may be repeated) and need to return combinations summing to target
// 1. similar to the prev sol except in helper func we recursively call for for helper func for idx <- idx+1 to n-1, to prevent repeated ele in subset
// 2. we also check in the for loop if a new element being added to subset is added only once - to prevent multiple repeated combinations
// example if vec = {1,1,2}, step 1 prevents [1,1,1] and step 2 prevents [[1,2](1; idx 0), [1,2](1: idx 1)]
// correct example for vec [2,5,2,1,2] and target 5 [[1,2,2],[5]]

// class Solution {
// public:
//     void helper(vector<int>& candidates, int target, int start,vector<int>& subset, vector<vector<int>>& ans){
//         if(target == 0){
//             ans.push_back(subset);
//             return;
//         }

//         for(int i = start; i < candidates.size(); i++){
//             if(target >= candidates[i]){
//                 subset.push_back(candidates[i]);
//                 helper(candidates, target-candidates[i], i+1, subset, ans);
//                 subset.pop_back();
//             }
//         }
//     }

//     vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
//         sort(candidates.begin(), candidates.end());
//         vector<vector<int>> ans;
//         vector<int> subset;
//         helper(candidates, target, 0, subset, ans);
//         return ans;
//     }
// };

class Solution {
public:
    // Helper function for combination sum using backtracking without reusing elements
    void helper(vector<int>& candidates, int target, vector<int>& subset, vector<vector<int>>& ans, int start) {
        if (target == 0) {
            ans.push_back(subset);
            return;
        }

        for (int i = start; i < candidates.size(); i++) {
            if (i > start && candidates[i] == candidates[i - 1]) continue; // Skip duplicates
            if (target >= candidates[i]) {
                subset.push_back(candidates[i]);
                helper(candidates, target - candidates[i], subset, ans, i + 1); // use i + 1 to avoid reusing elements
                subset.pop_back();
            }
        }
    }

    // Main function to find all unique combinations of candidates that sum to target without reuse
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> ans;
        vector<int> subset;
        helper(candidates, target, subset, ans, 0);
        return ans;
    }
};

// Q9 Subset sum
// find the sum of all subsets ex: [2,3] - [0,2,3,5]
// follow take and not take- pass the present idx and sum
void helper(vector<int> arr, int start, int sum, vector<int> & ans){
        if(start == arr.size()){
            ans.push_back(sum);
            return;
        }
            // not take
        helper(arr, start+1, sum, ans);
        // take
        helper(arr, start+1, sum+arr[start], ans);
    }
  
    vector<int> subsetSums(vector<int> arr, int n) {
        vector<int> ans;
        helper(arr, 0, 0, ans);
        return ans;
    }

// Q10
// power set - given vec may contain dup
// take not take method- time(100%): O(2^n) and space: O(n) - use of set - redundancy
class Solution {
public:
    void helper(vector<int>& nums, int start, vector<int>& subset, set<vector<int>>& ans){
        if(start == nums.size()){
            ans.insert(subset);
            return;
        }

        //not take
        helper(nums, start+1, subset, ans);
        //take
        subset.push_back(nums[start]);
        helper(nums, start+1, subset, ans);
        subset.pop_back();
        return;
    }

    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        set<vector<int>> ans;
        vector<int> subset;
        helper(nums, 0, subset, ans);
        vector<vector<int>> vc;
        vc.assign(ans.begin(), ans.end());
        return vc;
    }
};

// 
class Solution {
public:
    bool isdup(vector<vector<int>> &ans, vector<int>& x){
        for(vector<int>& y : ans){
            if(y.size() == x.size() && equal(y.begin(), y.end(), x.begin()))
            return true;
        }
        return false;
    }

    vector<vector<int>> helper(vector<int>& v, int t){
        if(t >= v.size()) return {{}};

        vector<vector<int>> partialans = helper(v, t+1);
        vector<vector<int>> ans;

        for(vector<int> x:partialans){
            ans.push_back(x);
        }
        for(vector<int> x:partialans){
            x.push_back(v[t]);
            if(!isdup(ans, x))
                ans.push_back(x);
            else x.pop_back();
        }
        return ans;
    }

    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        return helper(nums, 0);
    }
};

// Q11
// using only digits 0-9(no repeatition) return the subsets of len k summing to n
// time(100%): O(2^n), space: O(k*ans.size()) 
class Solution {
public:
    void helper(int start, int nums, int k, int sum, vector<int>& subset, vector<vector<int>>& ans){
        if(sum == 0 && nums == k)ans.push_back(subset);
        if(start > sum || start > 9) return;

        // not take
        helper(start+1, nums, k, sum, subset, ans);

        // take
        subset.push_back(start);
        helper(start+1, nums + 1, k, sum-start, subset, ans);
        subset.pop_back();
    }

    vector<vector<int>> combinationSum3(int k, int n) {
        vector<int> subset;
        vector<vector<int>> ans;
        helper(1, 0, k, n, subset, ans);
        return ans;
    }
};

// Q12 Phone letter combinations
// digits = "23" ; output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
// using f(n) = f(n-1) where n is the idx
// if we have the ans for f(n-1) we just need to the the chars correspoding to digits[n] to the end of all str in f(n-1)
// f(n-1) for prev ex is  ["a","b","c"] and corresponding chars to 3 is "def"
// need to optimise maybe by using idx from start - like in prev methods ;instead of calling partial ans
class Solution {
public:
    vector<string> letter(string digits, int start){
        if(start == -1) return {{}};
        vector<string> partialans = letter(digits, start-1);
        vector<string> ans;

        if(digits[start] == '2'){
            string str = "abc";
            for(int i = 0; i < str.size(); i++){
                for(auto s: partialans){
                    s.push_back(str[i]);
                    ans.push_back(s);
                    s.pop_back();
                } 
            }
        }
        else if(digits[start] == '3'){
            string str = "def";
            for(int i = 0; i < str.size(); i++){
                for(auto s: partialans){
                    s.push_back(str[i]);
                    ans.push_back(s);
                    s.pop_back();
                } 
            }
        }
        else if(digits[start] == '4'){
            string str = "ghi";
            for(int i = 0; i < str.size(); i++){
                for(auto s: partialans){
                    s.push_back(str[i]);
                    ans.push_back(s);
                    s.pop_back();
                } 
            }
        }
        else if(digits[start] == '5'){
            string str = "jkl";
            for(int i = 0; i < str.size(); i++){
                for(auto s: partialans){
                    s.push_back(str[i]);
                    ans.push_back(s);
                    s.pop_back();
                } 
            }
        }
        else if(digits[start] == '6'){
            string str = "mno";
            for(int i = 0; i < str.size(); i++){
                for(auto s: partialans){
                    s.push_back(str[i]);
                    ans.push_back(s);
                    s.pop_back();
                } 
            }
        }
        else if(digits[start] == '7'){
            string str = "pqrs";
            for(int i = 0; i < str.size(); i++){
                for(auto s: partialans){
                    s.push_back(str[i]);
                    ans.push_back(s);
                    s.pop_back();
                } 
            }
        }
        else if(digits[start] == '8'){
            string str = "tuv";
            for(int i = 0; i < str.size(); i++){
                for(auto s: partialans){
                    s.push_back(str[i]);
                    ans.push_back(s);
                    s.pop_back();
                } 
            }
        }
        else if(digits[start] == '9'){
            string str = "wxyz";
            for(int i = 0; i < str.size(); i++){
                for(auto s: partialans){
                    s.push_back(str[i]);
                    ans.push_back(s);
                    s.pop_back();
                } 
            }
        }
        return ans;
    }

    vector<string> letterCombinations(string digits) {
        if(digits.size() == 0)return {};
        return letter(digits, digits.size()-1);
    }
};


// 2 approaches for recursion
// finding f(n-1) and using it in f(n)
// iterating - take. not take


// -------------------------------------------HARD----------------------------------------------------------

// When you have many options with multiple possibilities like power set (take not take) etc or as in hard q2 - use recursion

// Q1
class Solution {
public:
    bool isPalindrome(string s){
        if(s.size() == 0)return true;
        for(int i = 0; i < s.size()/2; i++){
            if(s[i] != s[s.size()-1-i])return false;
        }
        return true;
    }

    void helper(const string& s, int idx, vector<string>& substr, vector<vector<string>>& ans) {
        if (idx == s.size()) {
            ans.push_back(substr);
            return;
        }
        
        for (int i = idx; i < s.size(); ++i) {
            string currentStr = s.substr(idx, i - idx + 1);
            if (isPalindrome(currentStr)) {
                substr.push_back(currentStr);
                helper(s, i + 1, substr, ans);
                substr.pop_back();
            }
        }
    }

    vector<vector<string>> partition(string s) {
        vector<vector<string>> ans;
        vector<string> substr;
        helper(s, 0, substr, ans);
        return ans;
    }
};

// Q2 
// given a mxn board with chars and a word and need to find if there is any path horz or vert to trace word 
// ex. [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]. word ="ABCCED" ans- true(rrrddl), rep of a pos not allowed
// find the see (word[0]) from there look for next char on r,l,u,d and repeat- each time when found mark the location, if path doesnt exist revert the org char (at this point we can also use for backtrack) 
class Solution {
public:
    int Search(vector<char> &vec, char c, int start){
        cout << start << endl;
        for(int i = start; i < vec.size(); i++){
            if(vec[i] == c){
                vec[i] = '.';
                return i;
            }
        }
        return -1;
    }

    bool helper(vector<vector<char>>& board, int row, int col, int idx, string word){
        if(idx == word.size())return true;
        // cout << row << " " << col << " " << word[idx-1] << endl;

        board[row][col] = '.';
        if(row > 0 && board[row-1][col] == word[idx]) {
            if(helper(board, row-1, col, idx+1, word)) return true;
        }
        if(row < board.size()-1 && board[row+1][col] == word[idx]){
            if (helper(board, row+1, col, idx+1, word)) return true;
        }
        if(col > 0 && board[row][col-1] == word[idx]) {
            if (helper(board, row, col-1, idx+1, word)) return true;
        }
        if(col < board[0].size()-1 && board[row][col+1] == word[idx]){
            if (helper(board, row, col+1, idx+1, word)) return true;
        }
        board[row][col] = word[idx-1];

        return false;
    }

    bool exist(vector<vector<char>>& board, string word) {
        int m = board.size(), n = board[0].size();
        char c = word[0];
        int i = 0;
        while(i < m){
            int col = Search(board[i], c, 0);
            while(col != -1){
                if(helper(board, i, col, 1, word)) return true;
                col = Search(board[i], c, col+1);
            }
            i++;
        }
        return false;
    }
};

// --------------------HARD-----------------------------------
// Q6 kth permutation sequence
// notice that n! had n (n-1)'s, so depending on the k permutation needed we can find out the 1st digit (i.e, the 1st (n-1) permutations have the smallest num as the leftmost digit and so on)
// in  that way, we r ble to reduce the num of digits and the num of permutation needed
// time(100%): O(n) {each recursion n dec by 1}, space: O(n) { vec and stack space}
class Solution {
public:
    int factorial(int i){
        if(i < 3)return i;
        else if (i == 3)return 6;
        else if (i == 4)return 24;
        else if (i == 5)return 120;
        else if (i == 6)return 720;
        else if (i == 7)return 5040;
        else if (i == 8)return 40320;
        else return 362880;
    }

    string helper(int n, int k, vector<int>order){
        if(n == 1) return to_string(order[0]);
        int fact = factorial(n-1);
        int q = k/fact;
        k = k%fact;

        string digit = to_string(order[q]);
        for(int i = q; i < n-1; i++){
            order[i] = order[i+1];
        }

        return (digit + helper(n-1, k, order));
    }

    string getPermutation(int n, int k) {
        vector<int>order;
        for(int i = 0; i < n; i++){
            order.push_back(i+1);
        }
        return helper(n, k-1, order);
    }
};

// Recursion and Backtracking
// Q1 All permutations
// have a start variable t denoting the idx till which permutation has occured, similar to tower of hanoi
class Solution {
public:
    void swap(int* a, int* b){
        int temp = *a;
        *a = *b;
        *b = temp;
    }

    void helper(vector<int> &v, int t, vector<vector<int>> &result){
        if(t == v.size()){
            result.push_back(v);
            return;
        }

        helper(v, t+1,result);
        for(int i = t+1; i < v.size(); i++){
            swap(&v[t], &v[i]);
            helper(v, t+1, result);
            swap(&v[t], &v[i]);
        }
        return;
    }

    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> result;
        helper(nums, 0, result);
        return result;
    }
};