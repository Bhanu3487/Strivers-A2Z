#include <iostream>
#include <bits/stdc++.h> 
using namespace std;

// lec 5: DP on Strings
// comaparision, replacement

// Q1. longest common subsequence  - DP-25
// brute force - recursion - generate all subsequences and check - exponential - O(2^n+ 2^m)

// 1. express in terms of idx (arr till idx a[0...idx]), 2. explore possibilities in that idx (take not take),3. text the optimal among them(max of take and not take)
// 1. in this q, we need 2 idxs - bcz of 2 strs
// 2. do characterwise conparision (of the cur idx) : match- reduce f(idx1, idx2) = f(idx1-1, idx2-1) + (str1[idx1] == str2[idx2]), other possibilities - not match : f(idx1-1, idx2), f(idx1, idx2-1), if the prev char of a str is matching the cur char of another str
// 3. take the max of all 

// 2D memoization (TLE)
// time: O(n*m), space:O(n*m)+O(n+m) <- n+m is auxilary stack space and should be optimised - each recursion we red n or m by 1
class Solution {
public:
    int func(int idx1, int idx2, string text1, string text2, vector<vector<int>>& dp){
        if(idx1 < 0 || idx2 < 0)return 0;
        if(dp[idx1][idx2] != -1)return dp[idx1][idx2];

        // match
        if(text1[idx1] == text2[idx2])return dp[idx1][idx2] =  1 + func(idx1-1, idx2-1, text1, text2, dp);
        // not match
        return dp[idx1][idx2] = max(func(idx1, idx2-1, text1, text2, dp), func(idx1-1, idx2, text1, text2, dp));
    }

    int longestCommonSubsequence(string text1, string text2) {
        int n = text1.size(), m = text2.size();
        vector<vector<int>> dp(n, vector<int>(m,-1));
        return func(n-1, m-1, text1, text2, dp);
    }
};

//shift of idxs
class Solution {
public:
    int func(int idx1, int idx2, const string &text1, const string &text2, vector<vector<int>>& dp) {
        if (idx1 == 0 || idx2 == 0) return 0;
        if (dp[idx1][idx2] != -1) return dp[idx1][idx2];

        if (text1[idx1 - 1] == text2[idx2 - 1]) return dp[idx1][idx2] = 1 + func(idx1 - 1, idx2 - 1, text1, text2, dp);
        
        return dp[idx1][idx2] = max(func(idx1, idx2 - 1, text1, text2, dp), func(idx1 - 1, idx2, text1, text2, dp));
    }
    
    int longestCommonSubsequence(const string &text1, const string &text2) {
        int n = text1.size(), m = text2.size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, -1));
        return  func(n, m, text1, text2, dp);
    }
};


// 2D tabulation
// time: O(n*m), space:O(n*m)
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
       int n = text1.size(), m = text2.size();
       vector<vector<int>> dp(n, vector<int>(m,0));

       for(int i = 0; i < m; i++) if(text1[0] == text2[i] || (i > 0 && dp[0][i-1] == 1)) dp[0][i] = 1;
       
       for(int i = 1; i < n; i++) if(text1[i] == text2[0] || dp[i-1][0] == 1) dp[i][0] = 1;
       
       for(int i = 1; i < n; i++){
            for(int j = 1; j < m; j++){
                dp[i][j] = max(dp[i-1][j-1]+(text1[i] == text2[j]), max(dp[i-1][j], dp[i][j-1]));
            }
       }

       return dp[n-1][m-1];
    }
};

// 2 1D arrays
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int n = text1.size(), m = text2.size();
        vector<int>prev(m,0), cur(m,0);

        for(int i = 0; i < m; i++) if(text1[0] == text2[i] || (i > 0 && prev[i-1] == 1))prev[i] = 1;
       
        for(int i = 1; i < n; i++){
            cur[0] = ((text1[i] == text2[0]) || prev[0] == 1);
            for(int j = 1; j < m; j++){
                cur[j] = max(prev[j-1]+(text1[i] == text2[j]), max(prev[j], cur[j-1]));
            }
            prev = cur;
            cout<<endl;
        }

        return prev[m-1];
    }
};

// Q2 - all largest common subsequences - DP-26
// 2D DP memoization - TLE - can be made better
class Solution{
public:
    int len = 0;
    int func(int idx1, int idx2, const string &text1, const string &text2, vector<vector<int>>& dp, vector<vector<string>>& dp2) {
        if (idx1 == 0 || idx2 == 0) return dp[idx1][idx2] = 0;
        if (dp[idx1][idx2] != -1) return dp[idx1][idx2];

        dp[idx1][idx2] = 0;

        int d = 0;
        if (text1[idx1-1] == text2[idx2-1]) {
            d = func(idx1 - 1, idx2 - 1, text1, text2, dp, dp2);
            dp[idx1][idx2] = 1 + d;
            dp2[idx1][idx2].push_back('D');
        }
        
        int l = func(idx1, idx2 - 1, text1, text2, dp, dp2);
        int u = func(idx1 - 1, idx2, text1, text2, dp, dp2);
        if(l >= u){
            if(l > d) dp[idx1][idx2] = l;
            if(l >= d) dp2[idx1][idx2].push_back('L');
        }
        if(u >= l){
            if(u > d) dp[idx1][idx2] = u;
            if(u >= d) dp2[idx1][idx2].push_back('U');
        }
        return dp[idx1][idx2];
    }
    
    void helper(const string &text1, int idx1, int idx2, string substr, set<string>& ans, vector<vector<string>>& dp2){
        if (idx1 == 0 || idx2 == 0) {
            if(substr.size() == len) {
                reverse(substr.begin(), substr.end());
                ans.insert(substr);
            }
            return;
        }
        
        for(int i = 0; i < dp2[idx1][idx2].size(); i++){
            char c = dp2[idx1][idx2][i];
            if (c == 'D') helper(text1, idx1-1, idx2-1, substr + text1[idx1-1], ans, dp2);
            else if (c == 'U') helper(text1, idx1-1, idx2, substr, ans, dp2);
            else if (c == 'L') helper(text1, idx1, idx2-1, substr, ans, dp2);
        }
        return;
    }
    
    vector<string> all_longest_common_subsequences(string text1, string text2){
        int n = text1.size(), m = text2.size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, -1));
        vector<vector<string>> dp2(n + 1, vector<string>(m + 1, ""));
        len = func(n, m, text1, text2, dp, dp2);
        set<string> ans;
        helper(text1, n, m, "", ans, dp2);
        
        return vector<string>(ans.begin(), ans.end());
    }
};

// 2D Tabulation
class Solution{
public:
    vector<string> all_longest_common_subsequences(string s, string t) {
        int n = s.size(), m = t.size();
        vector<vector<set<string>>> dp(n + 1, vector<set<string>>(m + 1));
    
        // Initialize dp with an empty string
        dp[0][0].insert("");
    
        for (int i = 1; i <= n; ++i) {
            dp[i][0].insert("");
        }
        for (int j = 1; j <= m; ++j) {
            dp[0][j].insert("");
        }
    
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m; ++j) {
                if (s[i - 1] == t[j - 1]) {
                    for (const string& str : dp[i - 1][j - 1]) {
                        dp[i][j].insert(str + s[i - 1]);
                    }
                } else {
                    if (dp[i - 1][j].begin()->size() >= dp[i][j - 1].begin()->size()) {
                        dp[i][j].insert(dp[i - 1][j].begin(), dp[i - 1][j].end());
                    }
                    if (dp[i][j - 1].begin()->size() >= dp[i - 1][j].begin()->size()) {
                        dp[i][j].insert(dp[i][j - 1].begin(), dp[i][j - 1].end());
                    }
                }
            }
        }
    
        return vector<string>(dp[n][m].begin(), dp[n][m].end());
    }
};

// Q3 - longest common substring

// can do recursion or memoization by keeping track of another variable
// 2D tabulation
class Solution{
    public:
    int longestCommonSubstr (string S1, string S2, int n, int m){
        vector<vector<int>> dp(n+1, vector<int> (m+1, 0));
        
        int ans = 0;
        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= m; j++){
                if(S1[i-1] == S2[j-1])dp[i][j] = dp[i-1][j-1] + 1;
                else dp[i][j] = 0;
                ans = max(ans, dp[i][j]);
            }
        }

        return ans;
    }
};


// 2 1D tabulation
class Solution{
    public:
    int longestCommonSubstr (string S1, string S2, int n, int m){
        vector<int> prev(n+1, 0), cur(m+1, 0);      // both are m-len of col
        
        int ans = 0;
        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= m; j++){
                if(S1[i-1] == S2[j-1])cur[j] = prev[j-1] + 1;
                else cur[j] = 0;
                ans = max(ans, cur[j]);
            }
            prev = cur;                             //imp dont forget
        }

        return ans;
    }
};

// Q4 - largest palindrome subsequence
// same as largest common subsequence, except the secend str is the rev one

// 2D tabulation
class Solution {
public:
    int longestPalindromeSubseq(string S1) {
        string S2 = S1; 
        std::reverse(S2.begin(), S2.end());        
        int n = S1.size();
        vector<vector<int>> dp(n+1, vector<int> (n+1, 0));
    
        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= n; j++){
                if(S1[i-1] == S2[j-1])dp[i][j] = 1+dp[i-1][j-1];
                else dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }

        return dp[n][n];
    }
};


// Q5 - minimum insersions to ma a str a palindrome
// maximum = len of str => str+ rev(str)

// DP approach - find the longest palindromic subsequence in the str and add the remaining ele to maintain palindrome

class Solution {
public:
    int minInsertions(string S1) {
        string S2 = S1; 
        std::reverse(S2.begin(), S2.end());        
        int n = S1.size();
        vector<int> prev(n+1, 0), cur(n+1, 0);
    
        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= n; j++){
                if(S1[i-1] == S2[j-1])cur[j] = 1+prev[j-1];
                else cur[j] = max(prev[j], cur[j-1]);
            }
            prev = cur;
        }
        int pal = cur[n];
        return S1.size()-pal;
    }
};

// Q6 - convert S1 to S2 by deleting chars
// find the lar com subseq and delete remaining char
class Solution {
public:
    int minDistance(string S1, string S2) {
        int n = S1.size(), m = S2.size();
        vector<int> prev(m+1, 0), cur(m+1, 0);
    
        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= m; j++){
                if(S1[i-1] == S2[j-1])cur[j] = 1+prev[j-1];
                else cur[j] = max(prev[j], cur[j-1]);
            }
            prev = cur;
        }
        int req = cur[m];
        return m+n-2*req;
    }
};

// Q9 print the shortest common superseq
// largest = str1+str2
// DP approach - finding the largest common subseq and adding the other chars - memory limit error
class Solution {
public:
    string shortestCommonSupersequence(string S1, string S2) {
        int n = S1.size(), m = S2.size();
        vector<string> prev(m+1, ""), cur(m+1, "");
    
        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= m; j++){
                if(S1[i-1] == S2[j-1])cur[j] = prev[j-1]+S1[i-1];
                else if(prev[j].size() > cur[j-1].size()) cur[j] = prev[j];
                else cur[j] = cur[j-1];
            }
            prev = cur;
        }
        string ans = "";
        string S = cur[m];
        char c1 = '.', c2 = '.', c = '.';
        int i = 0, j = 0, k = 0;
        while(k < S.size()){
            if(i < n) c1 = S1[i];
            if(j < m) c2 = S2[j];
            if(k < S.size()) c = S[k];

            if(c1 == c && c2 == c){ans.push_back(c); i++; j++; k++;}
            else if (c1 != c){ans.push_back(c1); i++;}
            else{ans.push_back(c2); j++;}
        }
        while(i < n) {ans.push_back(S1[i]); i++;}
        while(j < m) {ans.push_back(S2[j]); j++;}

        return ans;
    }
};

// with backtracking
class Solution {
public:
    string shortestCommonSupersequence(string S1, string S2) {
        int n = S1.size(), m = S2.size();
        vector<vector<int>> dp(n+1, vector<int>(m+1, 0));

        // Fill the dp table
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (S1[i-1] == S2[j-1]) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                } else {
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }

        // Backtrack to construct the shortest common supersequence
        string ans = "";
        int i = n, j = m;
        while (i > 0 && j > 0) {
            if (S1[i-1] == S2[j-1]) {
                ans.push_back(S1[i-1]);
                i--; j--;
            } else if (dp[i-1][j] > dp[i][j-1]) {
                ans.push_back(S1[i-1]);
                i--;
            } else {
                ans.push_back(S2[j-1]);
                j--;
            }
        }

        // Add remaining characters of S1 or S2
        while (i > 0) {
            ans.push_back(S1[i-1]);
            i--;
        }
        while (j > 0) {
            ans.push_back(S2[j-1]);
            j--;
        }

        // The constructed string is in reverse order, reverse it before returning
        reverse(ans.begin(), ans.end());
        return ans;
    }
};

// within the aboe for loops - mem exd err
class Solution {
public:
    string shortestCommonSupersequence(string S1, string S2) {
        int n = S1.size(), m = S2.size();
        vector<vector<string>> dp(n+1, vector<string>(m+1, ""));
        
        // Fill the dp table and construct the result string
        for (int i = 0; i <= n; ++i) {
            for (int j = 0; j <= m; ++j) {
                if (i == 0) {
                    dp[i][j] = S2.substr(0, j);
                } else if (j == 0) {
                    dp[i][j] = S1.substr(0, i);
                } else if (S1[i-1] == S2[j-1]) {
                    dp[i][j] = dp[i-1][j-1] + S1[i-1];
                } else {
                    if (dp[i-1][j].size() < dp[i][j-1].size()) {
                        dp[i][j] = dp[i-1][j] + S1[i-1];
                    } else {
                        dp[i][j] = dp[i][j-1] + S2[j-1];
                    }
                }
            }
        }
        
        return dp[n][m];
    }
};


// Q10 - num of different subsequences from s == t

// recursion: 1. express evtg in terms of idx1 and idx2 (since there re 2 str)
// 2. explore all possibilities 3. count all ways 4. base case

// 1. f(i, j) = num of subseq of t[0...j] in s[0...i]
// 2. if(t[j] == s[i])f(i,j) = f(i-1, j-1)+f(i-1, j) := f(i-1, j-1) taking the s[i] and t[j], f(i-1, j)- we chose all the occurances of t[j] == s[k], k < i. else f(i,j) = f(i-1, j).
// time: O(2^m + 2^n) auxillary stack space: O(m+n)

// 2d tabulation - tle
class Solution {
public:
    int numDistinct(string s, string t) {
        int n = t.size(), m = s.size();

        vector<vector<int>> dp(n+1, vector<int> (m+1, 0));

        for(int i = 1; i <= m; i++){
            if(s[i-1] == t[0]) dp[1][i] = dp[1][i-1] + 1;
            else dp[1][i] = dp[1][i-1];
        }    

        for(int i = 2; i <= n; i++){
            int prev = 0;
            for(int j = i; j <= m; j++){
                if(t[i-1] == s[j-1]) {dp[i][j] = dp[i-1][j-1] + dp[i][prev];prev = j;}
                else dp[i][j] = dp[i][j-1];
            }
        }
        return dp[n][m];
    }
};

// 2 1D tabulation
class Solution {
public:
    int numDistinct(std::string s, std::string t) {
        int n = t.size(), m = s.size();
        if (n > m) return 0;

        const int MOD = 1e9 + 7;
        
        vector<int> prev(m + 1, 0), cur(m + 1, 0);

        for (int i = 1; i <= m; i++) {
            if (s[i - 1] == t[0]) prev[i] = (prev[i - 1] + 1) % MOD;
            else prev[i] = prev[i - 1] % MOD;
        }
        if (n == 1) return prev[m];

        for (int i = 2; i <= n; i++) {
            int b = 0;
            for (int j = i - 1; j <= m; j++) {
                if (t[i - 1] == s[j - 1]) {
                    cur[j] = (prev[j - 1] + cur[b]) % MOD;
                    b = j;
                } else {
                    cur[j] = cur[j - 1] % MOD;
                }
            }
            prev = cur;
        }
        return cur[m];
    }
};

// no use of b bcz anyway cur[j] == cur[b]
class Solution {
public:
    int numDistinct(string s, string t) {
        int n = t.size(), m = s.size();
        if (n > m) return 0;

        const int MOD = 1e9 + 7;
        
        vector<int> prev(m + 1, 0), cur(m + 1, 0);

        for (int i = 1; i <= m; i++) {
            if (s[i - 1] == t[0]) prev[i] = (prev[i - 1] + 1) % MOD;
            else prev[i] = prev[i - 1] % MOD;
        }
        if (n == 1) return prev[m];

        for (int i = 2; i <= n; i++) {
            for (int j = i - 1; j <= m; j++) {
                if (t[i - 1] == s[j - 1]) cur[j] = (prev[j - 1] + cur[j-1]) % MOD;
                else cur[j] = cur[j - 1] % MOD;
            }
            prev = cur;
        }
        return cur[m];
    }
};

// can be optimised to 1D tabulation, if structured properly, i.e n=s.size() and t= m.size() etc

// Q11 - Edit distance - convert word1 to word2 by add, del and rep
// 2D tabulation time: O(n*m)100%, space: O(m*n)30%
class Solution {
public:
    int minDistance(string word1, string word2) {
        if(word1 == word2) return 0;
        int n = word1.size(), m = word2.size();

        vector<vector<int>> dp(n+1, vector<int>(m+1,0));

        for(int i = 1; i <= n; i++) dp[i][0] = i;
        for(int i = 1; i <= m; i++) dp[0][i] = i;

        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= m; j++){
                if(word1[i-1] == word2[j-1]) dp[i][j] = dp[i-1][j-1];
                else{
                    dp[i][j] = 1 + min(min(dp[i-1][j], dp[i][j-1]), dp[i-1][j-1]);
                }
            }
        }

        return dp[n][m];
    }
};

// 2 !d tabulation
// time : o(m*n)100 space: O(m) 95%
class Solution {
public:
    int minDistance(string word1, string word2) {
        if(word1 == word2) return 0;
        int n = word1.size(), m = word2.size();
        if(n <= 1 && m <= 1)return 1;

        vector<int> prev(m+1, 0), cur(m+1, 0);

        for(int i = 1; i <= m; i++) prev[i] = i;

        for(int i = 1; i <= n; i++){
            cur[0] = i;
            for(int j = 1; j <= m; j++){
                if(word1[i-1] == word2[j-1]) cur[j] = prev[j-1];
                else{
                    cur[j] = 1 + min(min(prev[j], cur[j-1]), prev[j-1]);
                }
            }
            prev = cur;
        }

        return cur[m];
    }
};

// 5, DP on stocks ----------------------------------------------------------
// Q1. best time to buy and sell stocks
// 2D tabluation - tle : time:O(n^2) space:O(n^2)
// 1D tabulation - tle : time:O(n^2) space:O(n)

// np dp sol'n
// time: O(n), space:O(1)
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();

        int m = INT_MAX, ans = INT_MIN;
        for(int i = 1; i < n; i++){
            m = min(prices[i-1], m);
            ans = max(prices[i] - m, ans);
        }

        if(ans < 0)return 0;
        return ans;
    }
};

// Q2. buy and sell stocks but only 1 stock at a time (can buy multiple times)

// recursion - tle - time: O(2^n), space: O(n) ASS
class Solution {
public:
    int helper(int idx, bool buy, vector<int>& prices){
        if(idx == prices.size())return 0; 
        int profit = 0;
        if(buy == 1){
            // buy & not buy
            profit = max(helper(idx+1, 0, prices) - prices[idx], helper(idx+1, 1, prices));
        }
        else{
            // sell & not sell
            profit = max(prices[idx] + helper(idx+1, 1, prices), helper(idx+1, 0, prices));
        }
        return profit;
    }

    int maxProfit(vector<int>& prices) {
        return helper( 0, 1, prices);
    }
};

// memoization: time:O(2n) 35%, space:O(2n)+O(n)ASS 16%
class Solution {
public:
    int helper(int idx, bool buy, vector<int>& prices, vector<vector<int>>& dp){
        if(idx == prices.size())return 0; 
        if(dp[idx][buy] != -1)return dp[idx][buy];
        int profit = 0;
        if(buy == 1){
            // buy & not buy
            profit = max(helper(idx+1, 0, prices, dp) - prices[idx], helper(idx+1, 1, prices, dp));
        }
        else{
            // sell & not sell
            profit = max(prices[idx] + helper(idx+1, 1, prices, dp), helper(idx+1, 0, prices, dp));
        }
        return dp[idx][buy] = profit;
    }

    int maxProfit(vector<int>& prices) {
        vector<vector<int>>dp(prices.size(), vector<int>(2,-1));
        return helper( 0, 1, prices, dp);
    }
};

// tabulation, time:O(2n) 90%, space:O(2n) 20%
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<vector<int>>dp(n+1, vector<int>(2,0));
        dp[n][0] = dp[n][1] = 0;

        for(int idx = n-1; idx >= 0; idx--){
            for(int buy = 0; buy < 2; buy++){
                int profit = 0;
                if(!buy) profit = max(prices[idx] + dp[idx+1][1], dp[idx+1][0]);
                else profit =  max(dp[idx+1][0] - prices[idx], dp[idx+1][1]);
                dp[idx][buy] = profit;
            }
        }
        return dp[0][1];
    }
};

// space optimization; time: O(2n) 90%, space: O(2*2) 40%
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<int> cur(2,0), prev(2,0);
        cur[0] = cur[1] = 0;

        for(int idx = n-1; idx >= 0; idx--){
            for(int buy = 0; buy < 2; buy++){
                int profit = 0;
                if(!buy) profit = max(prices[idx] + cur[1], cur[0]);
                else profit =  max(cur[0] - prices[idx], cur[1]);
                prev[buy] = profit;
            }
            cur = prev;
        }
        return prev[1];
    }
};

// no dp sol'n, time:O(n) 100%, space:O(1) 50%
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int net_profit = 0;
        for(int i = 1; i < prices.size(); i++){
            if(prices[i] > prices[i-1]) net_profit += prices[i] - prices[i-1];
        }
        return net_profit;
    }
};

// Q3- same as above but can only buy stocks at max 2 times
// recursion - 1 new param added, num of stockes brought - tle
class Solution {
public:
    int func(int idx, int num, bool buy, vector<int>& prices){
        if(idx == prices.size())return 0;
        if(num == 3) return 0;

        int profit = 0;
        if(buy){
            // buy or not buy
            profit = max(func(idx+1, num + 1, 0, prices) - prices[idx], func(idx+1, num, 1, prices));
        }
        else{
            // sell or not sell
            profit = max(func(idx + 1, num, 1, prices) + prices[idx], func(idx + 1, num, 0, prices));
        }

        return profit;
    }

    int maxProfit(vector<int>& prices) {
        return func(0, 0, 1, prices);
    }
}; 

// memoization: time:O(n*2*3), space:O(n*2*3) + O(n)ass
class Solution {
public:
    int func(int idx, int buy, int cap, vector<int>& prices, vector<vector<vector<int>>>& dp){
        if(idx == prices.size() || cap == 0)return 0;

        if(dp[idx][buy][cap] != -1) return dp[idx][buy][cap]; 
        if(buy == 1){
            // buy or not buy
            return dp[idx][buy][cap] = max( dp[idx+1][0][cap] - prices[idx], dp[idx+1][1][cap]);
        }
        
        return dp[idx][buy][cap] = max(dp[idx+1][1][cap-1] + prices[idx], dp[idx+1][0][cap]);
    }

    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<vector<vector<int>>> dp(n, vector<vector<int>>(2,vector<int>(3,-1)));

        return func(0, 1, 2, prices, dp);
    }
};

// tabulation time:O(n*2*3) 35%, space:O(N*2*3) 50%
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<vector<vector<int>>> dp(n+1, vector<vector<int>>(2,vector<int>(3,0)));
        
        for(int idx = n-1; idx >= 0; idx--){
            for(int buy = 0; buy <= 1; buy++){
                for(int cap = 0; cap <= 2; cap++){
                    if(cap == 0)dp[idx][buy][cap] = 0;
                    else{
                        if(buy == 1) dp[idx][buy][cap] = max( dp[idx+1][0][cap] - prices[idx], dp[idx+1][1][cap]);
                        else dp[idx][buy][cap] = max( dp[idx+1][1][cap-1] + prices[idx], dp[idx+1][0][cap]);
                    }
                }
            }
        }
        return dp[0][1][2];
    }
};

// space optimisation time:O(n*2*3)80%, space:O(2*3)90%
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<vector<int>> prev (2,vector<int>(3,0)), cur (2,vector<int>(3,0));
        
        for(int idx = n-1; idx >= 0; idx--){
            for(int buy = 0; buy <= 1; buy++){
                for(int cap = 0; cap <= 2; cap++){
                    if(cap == 0)prev[buy][cap] = 0;
                    else{
                        if(buy == 1) prev[buy][cap] = max( cur[0][cap] - prices[idx], cur[1][cap]);
                        else prev[buy][cap] = max( cur[1][cap-1] + prices[idx], cur[0][cap]);
                    }
                }
            }
            cur = prev;
        }
        return prev[1][2];
    }
};

// transaction method - tle
class Solution {
public:
    int func(int idx, int t, vector<int>& prices, vector<vector<int>>& dp){
        if(idx == prices.size() || t == 4)return 0;

        if(dp[idx][t] != -1) return dp[idx][t]; 
        if(t % 2 == 0){
            // buy or not buy
            return dp[idx][t] = max( func(idx+1,t+1,prices,dp) - prices[idx], func(idx+1,t,prices,dp));
        }
        //sell or not sell
        return dp[idx][t] = max(func(idx+1,t+1,prices,dp) + prices[idx], func(idx+1,t,prices,dp));
    }

    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<vector<int>>dp(n,vector<int>(4,-1));

        return func(0,0,prices,dp);
    }
};

// Q4 same as above but atmost k transactions
// time:O(n*2*k), space:O(n*2*k)+O(n)
class Solution {
public:
int func(int idx, int t, vector<int>& prices, vector<vector<int>>& dp){
        if(idx == prices.size() || t == dp[0].size())return 0;

        if(dp[idx][t] != -1) return dp[idx][t]; 
        if(t % 2 == 0){
            // buy or not buy
            return dp[idx][t] = max( func(idx+1,t+1,prices,dp) - prices[idx], func(idx+1,t,prices,dp));
        }
        //sell or not sell
        return dp[idx][t] = max(func(idx+1,t+1,prices,dp) + prices[idx], func(idx+1,t,prices,dp));
    }

    int maxProfit(int k, vector<int>& prices) {
        int n = prices.size();
        vector<vector<int>>dp(n,vector<int>(2*k,-1));

        return func(0,0,prices,dp);
    }
};

//tabulation, time:O(n*2*k)80% space:O(2*k*n)50%
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int n = prices.size();
        vector<vector<int>>dp(n+1,vector<int>(2*k+1,0));

        for(int idx = n-1; idx >= 0; idx--){
            for(int t = 2*k - 1; t >= 0; t--){
                if(t % 2 == 0) dp[idx][t] = max(dp[idx+1][t+1] - prices[idx], dp[idx+1][t]);
                else dp[idx][t] = max(dp[idx+1][t+1] + prices[idx], dp[idx+1][t]);
            }
        }
        return dp[0][0];
    }
};

// space optimization

// Q5 - q2 but after selling u have to wait for a day before buying - cooldown
// so thee are 3 states, buy,sell and cooldown
// if buy == 1: it means u can buy or not buy
// if but == 2: stock just sold so in cooldown, you can not  do anything but your state changes to 1 - i.e can buy next day
// if buy == 0: has brought a stock either sell or wait
class Solution {
public:
    int helper(int idx, int buy, vector<int>& prices, vector<vector<int>>& dp){
        if(idx == prices.size())return 0; 
        if(dp[idx][buy] != -1)return dp[idx][buy];
        int profit = 0;
        // 1: buy(1->0) & not buy(1->1)
        if(buy == 1)return dp[idx][buy] = max(helper(idx+1, 0, prices, dp) - prices[idx], helper(idx+1, 1, prices, dp));
        // 2: not but/not sell (change to 1)
        else if(buy == 2) return dp[idx][buy] = helper(idx+1, 1, prices, dp);
        // 0: sell(0->2) & not sell(0->0)
        else return dp[idx][buy] = max(prices[idx] + helper(idx+1, 2, prices, dp), helper(idx+1, 0, prices, dp));
    }

    int maxProfit(vector<int>& prices) {
        vector<vector<int>>dp(prices.size(), vector<int>(3,-1));
        return helper( 0, 1, prices, dp);
    }
};

// tabulation , time:O(n*3) 100%, space:O(n*3)70%
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<vector<int>>dp(n+1, vector<int>(3,0));

        for(int idx = n-1; idx >= 0; idx--){
            for(int buy = 0; buy < 3; buy++){
                if(buy == 1) dp[idx][buy] = max(dp[idx+1][0] - prices[idx], dp[idx+1][1]);
                else if(buy == 2) dp[idx][buy] = dp[idx+1][1];
                else dp[idx][buy] = max(prices[idx] + dp[idx+1][2], dp[idx+1][0]);
            }
        }
        return dp[0][1];
    }
};

//space optim- time:O(3n)100%. space:O(3*2)95%
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<int> prev(3,0), cur(3,0);

        for(int idx = n-1; idx >= 0; idx--){
            for(int buy = 0; buy < 3; buy++){
                if(buy == 1) prev[buy] = max(cur[0] - prices[idx], cur[1]);
                else if(buy == 2) prev[buy] = cur[1];
                else prev[buy] = max(prices[idx] + cur[2], cur[0]);
            }
            cur = prev;
        }
        return prev[1];
    }
};

// different method- once we sell- we can only buy from the day after tmr - hence after[1]
// time:O(2*(n+2))80, space:O(3*3)70
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<int> prev(2,0), cur(2,0), after(2,0);

        for(int idx = n-1; idx >= 0; idx--){
            for(int buy = 0; buy < 2; buy++){
                if(buy == 1) prev[buy] = max(cur[0] - prices[idx], cur[1]);
                else prev[buy] = max(prices[idx] + after[1], cur[0]);
            }
            after = cur;
            cur = prev;
        }
        return prev[1];
    }
};

class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int n = prices.size();
        vector<int> cur(2,0), prev(2,0);
        cur[0] = cur[1] = 0;

        for(int idx = n-1; idx >= 0; idx--){
            for(int buy = 0; buy < 2; buy++){
                int profit = 0;
                if(!buy) profit = max(prices[idx] + cur[1], cur[0]);
                else profit =  max(cur[0] - prices[idx] - fee, cur[1]);
                prev[buy] = profit;
            }
            cur = prev;
        }
        return prev[1];
    }
};

// Q6 - same q as q2 - but each transaction has fee - just charge the stock when brought 
class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int n = prices.size();
        vector<int> cur(2,0), prev(2,0);
        cur[0] = cur[1] = 0;

        for(int idx = n-1; idx >= 0; idx--){
            for(int buy = 0; buy < 2; buy++){
                int profit = 0;
                if(!buy) profit = max(prices[idx] + cur[1], cur[0]);
                else profit =  max(cur[0] - prices[idx] - fee, cur[1]); // only change in this line
                prev[buy] = profit;
            }
            cur = prev;
        }
        return prev[1];
    }
};

//
// Q1 
// recursion - 
class Solution {
public:
    int f(int idx, int num, vector<int>& nums){
        if(idx == -1) return 0;
        int take = 0;
        if(nums[idx] < nums[num]) take = f(idx-1, idx, nums)+1;
        int not_take = f(idx-1, num, nums);
        return max(take, not_take);
    }

    int lengthOfLIS(vector<int>& nums) {
        nums.push_back(INT_MAX);
        int n = nums.size();
        return f(n-2, n-1, nums);
    }
};

// tabulation
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        nums.push_back(INT_MAX);
        int n = nums.size();
        vector<vector<int>>dp(n, vector<int>(n,0));

        for(int idx = 0; idx < n-1; idx++){
            for(int num = 0; num < n; num++){
                int take = 0;
                if (nums[idx] < nums[num]) take = dp[idx-1][idx]+1;
                dp[idx][num] = max(dp[idx-1][num], take);
            }
        }
        return dp[n-2][n-1];
    }
};
