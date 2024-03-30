#include <iostream>

using namespace std;

// Q1 Assign cookies
// assign values from one vector s to another vector g 
// time: O(nlogn) space: O(1) 
class SolutionQ1 {
public:
    int findContentChildren(vector<int>& g, vector<int>& s) {
        int i = 0, j = 0;
        int ans = 0; 
        sort(g.begin(), g.end()); 
        sort(s.begin(), s.end()); 
        while(i < g.size() && j < s.size()){
            if(g[i] <= s[j]){
                ans++;
                i++; j++;
            }
            else {
                j++;
            }
        }
        return ans;
    }
};


//Q2 Fractional Knapsack ,ITEMS contains {weight, value} pairs.
// fill knapsack of capacity w weight with weights such that value is maximum,  fractional weights allowed
// time : O(nlogn)sort; space O(1)  
bool compareRatio(pair<int, int> a, pair<int, int> b) 
{ 
    double ratioA = (double)a.second / a.first;
    double ratioB = (double)b.second / b.first;
    return ratioA > ratioB; 
} 

double maximumValue(vector<pair<int, int>>& items, int n, int w)
{
    if (n == 0 || w == 0)
        return 0;

    sort(items.begin(), items.end(), compareRatio);

    double ans = 0.0;
    int currentWeight = 0;

    for (int i = 0; i < n; i++) {
        if (currentWeight + items[i].first <= w) {
            currentWeight += items[i].first;
            ans += items[i].second;
        } else {
            int remainingWeight = w - currentWeight;
            ans += (double)remainingWeight / items[i].first * items[i].second;
            break;
        }
    }
    return ans;
}

//Q3 min no. of coins
//given n should be split into currancy notes, we find by taking n/max_currancy, slowly decreasing max_currancy 
// time: O(n)?; space : O(1);
vector<int> MinimumCoins(int n)
{
    // Write your code here
    int i = 0;
    vector<int> ans;
    vector<int> coins{1000, 500, 100, 50, 20, 10, 5, 2, 1};
    while(i < coins.size()){
        while(n/coins[i] != 0){
            ans.push_back(coins[i]);
            n = n - coins[i];
        }
        i++;
    }
    return ans;
}

//Q4 lemonade change
// ans: can we give change to ith customer? simple keep count of 5's and 10's and return false for -ve/0 counts
//time: O(n), space: O(1)
class Solution {
public:
    bool lemonadeChange(vector<int>& bills) {
        int fives = 0, tens = 0;
        for (auto bill : bills) {
            
            if (bill == 5) 
                fives++;
            
            else if (bill == 10) {
                if (fives == 0) return false;
                tens++;
                fives--;
            }

            else {
                if (fives > 0 && tens > 0) {
                    fives--;
                    tens--;
                }
                else if (fives >= 3) fives -= 3;
                else return false;
            }
        }
        return true;
    }
};


// ----------------------------------------- MEDIUM ------------------------------------------

// Q2 JUMP GAME
// arr jump is given where from i we can jump max of jump[i] indices forward. can we reach the end?
// GREEDY sol'n: we can reach (n-1)th (last) pos if we can reach (n-2)th pos and jump[n-2] > 0.
// time: O(n) space = O(1)
bool jump_search(vector<int> &jump, int n){
    // Write your code here.
    int prez_idx = 0, aft_jump = 0; 
    while(prez_idx < n){
        aft_jump = prez_idx + jump[prez_idx];
        if(aft_jump >= n-1) return true;
    
        if(jump[prez_idx] > 0) prez_idx = aft_jump;
        else return false; 
    }
    return true;
}

// Q4 Minimum nmuber of platforms required
// sort arrival_time and departure time
// an incoming train cannot occupy a platform if ar_time < dp_time of the trains present so in that case we increase no. of platforms
// time: O(nlogn) for sorting; space : O(1)
int calculateMinPatforms(int at[], int dt[], int n) {
    vector<int> at_vect, dt_vect;
    for(int i = 0; i < n; i++){
        at_vect.push_back(at[i]);
        dt_vect.push_back(dt[i]);
    }

    sort(at_vect.begin(), at_vect.end()); sort(dt_vect.begin(), dt_vect.end());
    int a = 1, d = 0;
    int pt = 1, result = 1;

    while (a < n && d < n) {
        if (at_vect[a] <= dt_vect[d]){ pt++; a++; }
        else if(at_vect[a] > dt_vect[d]){ pt--; d++;}
        if(result < pt) result = pt;
    }
    return result;
}


// Q5 JOB SEQUENCING PROBLEM
// NX3 array jobs where; jobs[i][0] is id, jobs[i][1] is deadline, jobs[i][2] is profit for job
// Greedy sol'n : Create dp table storing jobs ids where idx is time(deadline)
// sort based on profits and then check for the highest idx available in dp_table and store its id
// time: O(nlogn) or maybe O(n**2)? ; space : O(n)
#include <algorithm>

bool compare_func(vector<int> &a, vector<int> &b){
    return a[2] > b[2];
}

vector<int> jobScheduling(vector<vector<int>> &jobs)
{
    vector<int> dp_table(jobs.size(), 0);
    sort(jobs.begin(), jobs.end(), compare_func);
    int max_profit = 0, num = 0;
    for(int i = 0; i < jobs.size(); i++){
        int j = jobs[i][2];
        int k = jobs[i][1] - 1;
        while(k >= 0){
          if (dp_table[k] == 0) {
            dp_table[k] = jobs[i][0];
            max_profit += j;
            num ++;
            break;
          }
          k--;
        }
    }
    return {num, max_profit};
}