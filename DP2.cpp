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

