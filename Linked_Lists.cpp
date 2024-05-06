#include <iostream>
#include <bits/stdc++.h>
using namespace std;

// ---------------------------1D LL---------------------------
// Q1. arr to ll 
// /**
//  * Definition of linked list
//  * class Node {
//  *
//  * public:
//  *     int data;
//  *     Node* next;
//  *     Node() : data(0), next(nullptr) {}
//  *     Node(int x) : data(x), next(nullptr) {}
//  *     Node(int x, Node* next) : data(x), next(next) {}
//  * };
//  */

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
/**
 * Definition of linked list
 * class Node {
 *
 * public:
 *     int data;
 *     Node* next;
 *     Node() : data(0), next(nullptr) {}
 *     Node(int x) : data(x), next(nullptr) {}
 *     Node(int x, Node* next) : data(x), next(next) {}
 * };
 */

Node* insertAtFirst(Node* list, int newValue) {
    Node *new_head = new Node(newValue);
    new_head->next = list;
    return new_head;
}

//Q3 Delete the given node
//put next node value in given node and remove the next node by updating the next of given node
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
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
/****************************************************************

    Following is the class structure of the Node class:

        class Node
        {
        public:
            int data;
            Node *next;
            Node(int data = 0, Node* next = NULL)
            {
                this->data = data;
                this->next = next;
            }
        };


*****************************************************************/

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
/****************************************************************

    Following is the class structure of the Node class:

    template <typename T>
    class Node
    {
    public:
        T data;
        Node<T> *next;
        Node(T data = 0, T* next = NULL)
        {
            this->data = data;
            this->next = next;
        }
    };

*****************************************************************/

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
/*
 * Definition for doubly-linked list.
 * class Node
 * {
 * public:
 *    int data;
 *    Node *next, *prev;
 *    Node() : data(0), next(nullptr), prev(nullptr) {}
 *    Node(int x) : data(x), next(nullptr), prev(nullptr) {}
 *    Node(int x, Node *next, Node *prev) : data(x), next(next), prev(prev) {}
 * };
 */

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
/**
 * Definition of doubly linked list:
 *
 * struct Node {
 *      int value;
 *      Node *prev;
 *      Node *next;
 *      Node() : value(0), prev(nullptr), next(nullptr) {}
 *      Node(int val) : value(val), prev(nullptr), next(nullptr) {}
 *      Node(int val, Node *p, Node *n) : value(val), prev(p), next(n) {}
 * };
 *
 *************************************************************************/

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
/**
 * Definition of doubly linked list:
 *
 * struct Node {
 *      int data;
 *      Node *prev;
 *      Node *next;
 *      Node() : data(0), prev(nullptr), next(nullptr) {}
 *      Node(int val) : data(val), prev(nullptr), next(nullptr) {}
 *      Node(int val, Node *p, Node *n) : data(val), prev(p), next(n) {}
 * };
 *
 *************************************************************************/

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
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
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
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
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

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
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
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
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
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
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

// 
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* oddEvenList(ListNode* head) {
        if(head == NULL || head->next == NULL) return head;
        ListNode* Odd = head;
        ListNode* ptr_node = head->next;
        ListNode* Even = head->next;
        while(true){
            if(Even->next != NULL){
                Odd->next = Even->next;
                Odd = Odd->next;
            }
            else break;
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