#include <iostream>
#include <bits/stdc++.h>
using namespace std;

//  Definition for singly-linked list.
  struct ListNode {
      int val;
      ListNode *next;
      ListNode() : val(0), next(nullptr) {}
      ListNode(int x) : val(x), next(nullptr) {}
      ListNode(int x, ListNode *next) : val(x), next(next) {}
  };


 // Definition of linked list
  class Node {
 
  public:
      int data;
      Node* next;
      Node() : data(0), next(nullptr) {}
      Node(int x) : data(x), next(nullptr) {}
      Node(int x, Node* next) : data(x), next(next) {}
 };

 // Definition for doubly-linked list.
  class Node
  {
  public:
     int data;
     Node *next, *prev;
     Node() : data(0), next(nullptr), prev(nullptr) {}
     Node(int x) : data(x), next(nullptr), prev(nullptr) {}
     Node(int x, Node *next, Node *prev) : data(x), next(next), prev(prev) {}
 };

// ---------------------------1D LL---------------------------
// Q1. arr to ll 

Node* constructLL(vector<int>& arr) {
    int n = arr.size();
    Node *temp = new Node(arr[0]);
    Node *head = temp;
    for(int i = 1; i < n; i++){
        temp->next =  new Node(arr[i]);
        temp = temp->next;
    }
    return head;
}

//Q2 insert node at head
Node* insertAtFirst(Node* list, int newValue) {
    Node *new_head = new Node(newValue);
    new_head->next = list;
    return new_head;
}

//Q3 Delete the given node
//put next node value in given node and remove the next node by updating the next of given node

class Solution {
public:
    void deleteNode(ListNode* node) {
        ListNode* nextNode = node->next;
        node->val = nextNode->val;
        node->next = nextNode->next;
        delete nextNode;
    }
};

// Q4 find len of ll

int length(Node *head)
{
    if(head == nullptr) return 0;
    Node *temp = head;
    int len = 1;
    while(temp->next != nullptr){
        len++;
        temp = temp->next;
    }
    return len;
}

//Q5 Search element Node<int>* temp = head;: similar to vector<int>

    // Following is the class structure of the Node class:

    // template <typename T>
    // class Node{
    // public:
    //     T data;
    //     Node<T> *next;
    //     Node(T data = 0, T* next = NULL){
    //         this->data = data;
    //         this->next = next;
    //     }
    // };

int searchInLinkedList(Node<int> *head, int k) {
    Node<int>* temp = head;
    while(temp != NULL){
        if(temp->data == k) return 1;
        temp = temp->next;
    }
    return 0;
}

//-----------------------2D LL----------------------------------
// Q1 Doubly LL
// have the current and the next node in loop update the prev of next and the next of current

Node* constructDLL(vector<int>& arr) {
    int n = arr.size();
    if (n == 0) return nullptr; 
    
    Node *head = new Node(arr[0]); 
    Node *temp = head;
    
    for(int i = 1; i < n; i++) {
        Node *newNode = new Node(arr[i]); 
        
        temp->next = newNode; 
        newNode->prev = temp;
        
        temp = newNode; 
    }
    
    return head; 
}


// Q2 insert at head : think of edge cases - empty ll

Node *insertAtTail(Node *head, int k) {
    Node* last = new Node(k);
    if(head == NULL) return last;
    Node *temp = head;
    while(temp->next != NULL){
        temp = temp->next;
    }

    temp->next = last;
    last->prev = temp;
    return head;
}

// Q3 Delete last node in DLL

Node * deleteLastNode(Node *head) {
    if(head == NULL || head->next == NULL) return NULL;
    Node* temp = head;
    while(temp->next != NULL){
        temp = temp->next;
    }
    temp->prev->next = NULL;
    return head;
}

// Q4 Reverse DLL
// with temp having next pointed to its prev and its prev pointed to next, now make the Next_Node also similar to temp and update temp to Next_Node
/*
Following is the class structure of the Node class:

class Node
{
public:
    int data;
    Node *next,*prev;
    Node()
    {
        this->data = 0;
        next = NULL;
        prev= NULL;
    }
    Node(int data)
    {
        this->data = data; 
        this->next = NULL;
        this->prev= NULL;
    }
    Node(int data, Node* next, Node *prev)
    {
        this->data = data;
        this->next = next;
        this->prev= prev;
    }
};

*/

Node* reverseDLL(Node* head)
{   
    if(head == NULL || head->next == NULL) return head;
    Node *temp = head;
    temp->prev = temp->next;
    temp->next = NULL;
    temp = head;
    while(temp->prev != NULL){
        Node* New_node = temp->prev;
        New_node->prev = New_node->next;
        New_node->next = temp;
        temp = New_node;
    }
    return temp;
}

// ----------------------------------MEDIUM SLL----------------------------
// Q1 - return the middle element
// if len == even return the 2nd middle ele
//time: O(n)- 2 passes and space : O(1)

class Solution {
public:
    ListNode* middleNode(ListNode* head) {
       if(head == NULL || head->next == NULL)return head;
       ListNode *temp = head;
       int len = 0;
       while(temp != NULL){
        len++;
        temp = temp->next;
       } 
       temp = head;
       for(int i = 0; i < (len/2); i++){
            temp = temp->next;
       }
       return temp;
    }
};

//Better method: Tortoise and Hare
//slow ptr inc by 1, fast ptr inc by 2, so by the time fast reaches the end, slow reaches the middle
//time:O(n), space: O(1)

class Solution {
public:
    ListNode* middleNode(ListNode* head) {
       if(head == NULL || head->next == NULL)return head;
       ListNode *slow = head;
       ListNode *fast = head;
       while(fast != NULL && fast->next != NULL){
            slow = slow->next;
            fast = fast->next->next;
       }
       return slow;
    }
};


// Q2 Reverse a SLL
// current has the node whose next we are changing to prev i.e temp, till temp it is reversed. save NextNode of current as we will lose it;
//time: O(n); space : O(1)

class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(head == NULL || head->next == NULL)return head;
        ListNode* temp = head;
        ListNode* current = head->next;
        temp->next = NULL;
        while(current != NULL){
            ListNode* NextNode = current->next;
            current->next = temp;
            temp = current;
            current = NextNode;
        }
        return temp;
    }
};


// Q4 Detect a cycle
// marking all the visited nodes by f(x) = (x>=0)?(x+10^5+1):(x-10^5) : mapping it to an int outside the range of val defined in question so they can be reverted back
// if anynum outside the range is found - loop exists

class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(head == NULL || head->next == NULL) return false;
        ListNode *temp = head;
        while(temp->val <= 100000 && temp->val >= -100000){
            int data = temp->val;
            if(data >= 0) temp->val += 100001;
            else temp->val -= 100000;
            temp = temp->next;
            if(temp == NULL) return false;
        }
        return true;
    }
};

//Another method (better): Tortoise and Hare
// slow pts inc by 1 and fast ptr inc by 2, in a loop they will definetely meet, if not a loop fast reaches/crosses the end first

class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(head == NULL || head->next == NULL) return false;
        ListNode *slow = head;
        ListNode *fast = head;
        while(fast != NULL && fast->next != NULL){
            slow = slow->next;
            fast = fast->next->next;
            if(fast == slow) return true;
        }
        return false;
    }
};

//Q8 Segregate Even and Odd nodes
 // create 2 ptr to nodes, 1. to track all odd nodes othere for even, the last of odds points to the 1st of even 
 // also need to save the 1st odd(head already in q) and 1st even(create a ptr) to which last odd points
 // odd node gets the next of even and odd gets updates, similarly for even, exit when any of odd or even next step is to become null
 //time: O(n), space: O(1)
class Solution {
public:
    ListNode* oddEvenList(ListNode* head) {
        if(head == NULL || head->next == NULL) return head;
        ListNode* Odd = head;
        ListNode* ptr_node = head->next;
        ListNode* Even = head->next;
        while(Even->next != NULL){
            Odd->next = Even->next;
            Odd = Odd->next;
            if(Odd->next != NULL){
                Even->next = Odd->next;
                Even = Even->next;
            }
            else {
                Even->next = NULL;
                break;
            }
        }
        Odd->next = ptr_node;
        return head;
    }
};

//same logic cleaner code

class Solution {
public:
    ListNode* oddEvenList(ListNode* head) {
        if(head == NULL || head->next == NULL) return head;
        ListNode* Odd = head;
        ListNode* ptr_node = head->next;
        ListNode* Even = head->next;
        while(Even != NULL && Even->next != NULL){
            Odd->next = Odd->next->next;
            Even->next = Even->next->next;
            Odd=Odd->next;
            Even=Even->next;
        }
        Odd->next = ptr_node;
        return head;
    }
};

// Q9 Delete Nth node from end
// brute force : 1 O(n) pass to end to find the len, then traverse len-n-1 and delete node - 2 passes
// optimal: jump start to fast ptr by n (fast covers n), then slow and fast inc by 1 (slow and fast cover len-n)
// due to some issues with edge cases - we add an extra node at the start
// time: O(n) 1 pass, space: O(1)

class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        if(head == NULL || head->next == NULL) return NULL;
        ListNode* start = new ListNode(0, head);
        ListNode* fast = start;
        for(int i = 0; i < n; i++){
            fast = fast->next;
            cout << i << fast->val << endl;
        }
        ListNode* slow = start;
        while(fast->next != NULL){
            slow = slow->next;
            fast = fast->next;
            // cout << slow->val << endl;
        }
        cout << slow->val;
        slow->next = slow->next->next;
        if(slow->val == 0)return start->next;
        return head;
    }
};

// Q10 Delete middle node
// using tortoise hare to find the middle node(but here we find middle+1 node)
// Dummy variable approach: so we add 1 node at start(even odd difference) and give fast a head start of 1  (balances the even odd diff)
// time: O(n) 1 pass, space: O(1)
class Solution {
public:
    ListNode* deleteMiddle(ListNode* head) {
        if(head == NULL || head->next == NULL)return NULL;
        ListNode* start = new ListNode(0, head);
        head = start;
        ListNode* fast = head->next;
        ListNode* slow = head;
        while(fast != NULL && fast->next != NULL){
            slow = slow->next;
            fast = fast->next->next;
        }
        slow->next = slow->next->next;
        return start->next;
    }
};

// by striver - no use of dummy variable
// problem we need to have fast lead by 2, so give it a head start by 2

class Solution {
public:
    ListNode* deleteMiddle(ListNode* head) {
        if(head == NULL || head->next == NULL)return NULL;
        ListNode* slow = head;
        ListNode* fast = head->next->next;
        while(fast != NULL && fast->next != NULL){
            slow = slow->next;
            fast = fast->next->next;
        }
        slow->next = slow->next->next;
        return head;
    }
};

// Q11 SortLL
// merging ll using merge sort
// didnt run
// sortll func: time:O(logN * (N+(N/2)) where height = logN and at each height N/2 to find middle and N comparisions in merge
//             space: O(1)
// in array time:O(NlogN) space:O(N)
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        if(head == NULL || head->next == NULL)return NULL;
        ListNode* middle = find_middle(head);
        ListNode* right = middle->next;
        middle->next = NULL;
        ListNode* left = head;

        left = sortList(left);
        right = sortList(right);
        return merge(left, right);
    }

    ListNode* find_middle(ListNode* head){
        if(head == NULL || head->next == NULL) return head;
        ListNode* slow = head;
        ListNode* fast = head;
        while(fast != NULL && fast->next != NULL){
            slow = slow->next;
            fast = fast->next->next;
        }
        return slow;
    }

    ListNode* merge(ListNode* list1, ListNode* list2){
        ListNode* dummy = new ListNode(-1);
        ListNode* temp = dummy;
        while(list1 != NULL && list2 != NULL){
            if(list1->val < list2->val){
                temp->next = list1;
                temp = list1;
                list1 = list1->next;
            }
            else{
                temp->next = list2;
                temp = list2;
                list2 = list2->next;
            }
        }
        if(list1) temp->next = list1;
        else temp->next = list2;
        return dummy->next;
    }
};


// Q12  Sort LL of 0, 1, 2
// if the next node is :
// 1: skip,  0: remove it and place it at head, 2: remove it and place at end
// if the ll starts with 2, then start by removing it and placing at end
// time: O(n), space: O(1)
Node* sortList(Node *head){
    Node* temp = new Node();
    Node* end = new Node();
    end = NULL;
    Node* cur = head;
    while(cur->data == 2){
        temp = cur->next;
        cur->next = end;
        end = cur;
        head = temp;
        cur = head;
    }
    while(cur->next != NULL){
        if(cur->next->data == 1){cur = cur->next;}
        else{
            temp = cur->next;
            cur->next = temp->next;
            if(temp->data == 0) {
                temp->next = head;
                head = temp;
            } else if (temp->data == 2) {
                temp->next = end;
                end = temp;
            }
        }
    }
    if (end != NULL) {
        cur->next = end;
    }
    return head;
}

// Equivalent - store all 0,1, and 2's separately in 3 ll by having 3 dummy nodes

// Q13 Find the intersection node
// 2 different ll join at a node and ot return the idx of this intersection
// brute force: store the 'nodes' (not data in nodes) of the 1st ll in a hashmap and then check each node of 2nd ll if it is in map
// time: O(n+m) space:O(n) - goal: reduce space

// method 2
// find the len of both the lls
// jump start to the longer ll and start comparing nodes 
// time: O(2*n + m) , space: O(1) - goal; reduce time
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(headA == NULL|| headB == NULL) return NULL;
        ListNode *t1 = headA, *t2 = headB;
        int lenA = 1;
        int lenB = 1;
        while(t1 != NULL){
            lenA++;
            t1 = t1->next;
        }
        while(t2 != NULL){
            lenB++;
            t2 = t2->next;
        } 
        ListNode *temp = headB;
        ListNode *other = headA;
        if(lenA > lenB){
            temp = headA;
            other = headB;
        }
        int len = abs(lenA - lenB);
        for(int i = 0; i < len; i++){
            temp = temp->next;
        }
        while(temp != NULL){
            if(temp == other) return temp;
            temp = temp->next;
            other = other->next;
        }
        return NULL;
    }
};

// Optimal
// have 2 pts traverse both ll, once one of them reaches end pnt it to the next ll and traverse till they reach
// they will collide at the next intersection (2nd pass) else they will collide at null
// time:O(n+m) space: O(1)
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(headA == NULL|| headB == NULL) return NULL;
        ListNode *t1 = headA, *t2 = headB;
        ListNode *temp = new ListNode();
        int flag1 = 1, flag2 = 1;
        while(t1 != t2 ){
            t1 = t1->next;
            t2 = t2->next;
            if(t1 == NULL && flag1){t1 = headB; flag1--;}
            if(t2 == NULL && flag2){t2 = headA; flag2--;}
        }
        if(t1 == NULL) return NULL;
        cout << t1->val << t2->val;
        return t1;
    }
};

// Add 1 to ll number head ptng to msb
// to add 1 at tail and return head(msb)
// q: 1 -> 5 -> 2. a:1 -> 5 -> 3 and 9 -> 9. a:1 -> 0 -> 0.
// reverse the ll, add 1 and reverse it back
// time: O(3n), space: O(1) - use like 4 extra nodes

Node *addOne(Node *head)
{
    Node * prev = head;
    Node * cur = head->next;
    Node * NextNode = NULL;
    prev->next = NULL;
    while(cur->next != NULL){
        NextNode = cur->next;
        cur->next =  prev;
        prev = cur;
        cur = NextNode;
    }
    cur->next = prev;


    head = cur;
    while(cur->data == 9 ){
        cur->data = 0;
        if(cur->next == NULL){
            Node * node = new Node(0);
            cur->next = node;
        }
        cur = cur->next;
    }
    cur->data += 1;


    prev = head;
    cur = head->next;
    NextNode = NULL;
    prev->next = NULL;
    while(cur->next != NULL){
        NextNode = cur->next;
        cur->next =  prev;
        prev = cur;
        cur = NextNode;
    }
    cur->next = prev;

    
    return cur;
}

// Q15 Add 2 numbers in LL
// Dummy variable method
// we store the sum in l2 and keep directing our dummy ll to it till it ends, then we update the sum in l1 and direct dummy to it
// create a node for the last carry if exists
// time: O(n1 + n2) space: O(1) // not very time efficient but memory efficient - other method just create a complete new ll saves times (less if cond)
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode* dummy = new ListNode(0);
        ListNode* t1 = l1;
        ListNode* t2 = l2;
        ListNode* cur = dummy;
        int sum = 0, carry = 0;
        while(t1 != NULL || t2 != NULL){
            sum = carry;
            if(t1 != NULL){
                sum += t1->val;
            }
            if(t2 != NULL){
                sum += t2->val;
                t2->val = sum % 10;
                carry = sum / 10;
                cur->next = t2;
            }
            if(t2 == NULL){
                t1->val = sum % 10;
                carry = sum / 10;
                cur->next = t1;
            }
            if(t1 != NULL) t1 = t1->next;
            if(t2 != NULL) t2 = t2->next;
            cur = cur->next; 
        }
        if(carry) {
            ListNode* node = new ListNode(1);
            cur->next = node;
        }
        return dummy->next;
    }
};

//-----------------------------------------2D LL MEDIUM-------------------------------------

// Q1 delete all nodes == to k
// check for edge cases- head and tail
// time:O(n), space: O(1)
void deleteAllOccurOfX(struct Node** head_ref, int x) {
        if (*head_ref == NULL) return;
        Node* temp = *(head_ref);
        while(temp -> next != NULL){
            if(temp->data == x){
                if (temp->prev == NULL) {
                    temp->next->prev = NULL;
                    *head_ref = temp->next;
                }
                else{
                    temp->prev->next = temp->next;
                    temp->next->prev = temp->prev;
                }
                temp = temp->next;
            }
            else temp = temp->next;
        }
        if(temp->data == x){
            temp->prev->next = NULL;
        }
        return;

// Q2
// find pair with given sum - sorted arr
// have a ptr at start and at end 
// not running for some reason

vector<pair<int, int>> findPairsWithGivenSum(Node *head, int target)
{
    vector<pair<int, int>> ans;
    Node *left = head;
    Node *right = head;
    while(right->next != NULL) right = right->next;
    while(left->prev != right){
        int sum = left->data + right->data;
        if(sum == target) {
            pair<int, int> PAIR1;
            PAIR1.first = left->data;
            PAIR1.second = right->data;
            ans.push_back(PAIR1);
            left = left->next;
            right = right->prev;
        }
        else if(sum < target) left = left->next;
        else right = right->prev;
    }
    return {};
}

// Q3 remove duplicates - sorted arr
Node * removeDuplicates(struct Node *head)
{
    Node *temp = head;
    while(temp -> next && temp->next != NULL){
        Node *nextNode = temp->next;
        while(nextNode != NULL && temp->data == nextNode->data){
            nextNode = nextNode->next;
        }
        temp->next = nextNode;
        if(nextNode != NULL) nextNode->prev = temp;
    }
    return head;
}


// ----------------------------------MEDIUM----------------------------------------


