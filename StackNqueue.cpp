#include <iostream>
using namespace std;
#include <vector>

// ----------------------------------LEARNING---------------------------------------
// 1. Implement Stack using queue
// Stack class.
class Stack {
private:
    int* arr;
    int capacity;
    int top_idx;

public:
    
    Stack(int capacity) {
        // Write your code here.
        this->capacity = capacity;
        arr = new int[capacity];
        top_idx = -1;
    }

    void push(int num) {
        // Write your code here.
        if(!isFull()){
            top_idx++;
            arr[top_idx] = num;
        }
    }

    int pop() {
        // Write your code here.
        if(!isEmpty()) {
            int ele = arr[top_idx];
            top_idx--;
            return ele;
        }else {
            return -1;
        }
    }
    
    int top() {
        // Write your code here.
        if(!isEmpty()) return arr[top_idx];
        else return -1;
    }
    
    int isEmpty() {
        // Write your code here.
        return (top_idx == -1);
    }
    
    int isFull() {
        // Write your code here.
        return (top_idx == capacity-1);
    }
    
};

// Implement Queue using Array
class Queue {

	int front, rear;
	vector<int> arr;

public:
	Queue()
	{
		front = 0;
		rear = 0;
		arr.resize(100001);
	}

	// Enqueue (add) element 'e' at the end of the queue.
	void enqueue(int e)
	{
		// Write your code here.
		if(rear < arr.size()){
			arr[rear] = e;
			rear++;
		}
	}

	// Dequeue (retrieve) the element from the front of the queue.
	int dequeue()
	{
		// Write your code here.
		if(front < rear){
			int ele = arr[front];
			front++;
			return ele;
		}else return -1;
	}
};

// Q3 stack using queue
#include <queue>
#include <stdexcept>
using namespace std;

class MyStack {
private:
    queue<int> q1;     // First queue used for stack operations
    queue<int> q2;     // Second queue used as a temporary helper
    int topElement;    // Stores the most recent element pushed to the stack

public:
    // Constructor: Initializes an empty stack
    MyStack() {
        // The constructor does not need to initialize anything explicitly
        // Default member initialization is sufficient
    }

    void push(int x) {
        q2.push(x);
        topElement = x;  // Update the top element

        // Move all elements from q1 to q2
        while (!q1.empty()) {
            q2.push(q1.front());
            q1.pop();
        }

        // Swap q1 and q2
        q1.swap(q2);
    }

    int pop() {
        if (q1.empty()) {
            throw runtime_error("Stack is empty");
        }

        int result = q1.front();
        q1.pop();

        if (!q1.empty()) {
            topElement = q1.front();
        }

        return result;
    }

    int top() {
        if (q1.empty()) {
            throw runtime_error("Stack is empty");
        }

        return topElement;
    }

    bool empty() {
        return q1.empty();
    }
};

// Q4 queue using stack
class MyQueue {
private:
    stack<int> s1;
    stack<int> s2;

public:
    MyQueue() {
        
    }
    
    void push(int x) {
        while(!s1.empty()){
            s2.push(s1.top());
            s1.pop();
        }
        s2.push(x);
        while(!s2.empty()){
            s1.push(s2.top());
            s2.pop();
        }
    }
    
    int pop() {
        int ele = s1.top();
        s1.pop();
        return ele;
    }
    
    int peek() {
        return s1.top();
    }
    
    bool empty() {
        return s1.empty();
    }
};

// Q5 Stack using LL
// void MyStack ::push(int x) 
// {
//     StackNode* node = new StackNode(x);
//     node->next = top;
//     top = node;
// }

// //Function to remove an item from top of the stack.
// int MyStack ::pop() 
// {
//     if(top == NULL)return -1;
//     int top_data = top->data; 
//     top = top->next;
//     return top_data;
// }

void MyStack ::push(int x) 
{
    StackNode* node = new StackNode(x);
    node->next = top;
    top = node;
}

//Function to remove an item from top of the stack.
int MyStack ::pop() 
{
    if(top == NULL)return -1;
    int top_data = top->data; 
    top = top->next;
    return top_data;
}

// Q6 Queue using LL
/* And structure of MyQueue
struct MyQueue {
    QueueNode *front;
    QueueNode *rear;
    void push(int);
    int pop();
    MyQueue() {front = rear = NULL;}
}; */

//Function to push an element into the queue.
void MyQueue:: push(int x)
{
    // QueueNode* node = new QueueNode(x);
    // node->next = rear;
    // rear = node;
    // if(rear->next == NULL || rear->next->next == NULL) front = rear; 
    
    QueueNode* node = new QueueNode(x);
    if (rear == NULL) {  
        front = rear = node;  
    } else {
        rear->next = node;  
        rear = node;        
    }
}

int MyQueue :: pop()
{
    // if(front->next == NULL)return -1;
    // int front_data = front->next->data;
    // front->next = NULL;
    // return front_data;
    
    if (front == NULL) return -1;
    int front_data = front->data;
    front = front->next;
    if (front == NULL) {  // Queue is now empty
        rear = NULL;
    }
    return front_data;
}

// Q7 Valid Parenthesis
class Solution {
public:
    bool isValid(string s) {
        stack<char> s1;
        char c = '.';
        for(int i = 0; i < s.size(); i++){
            if(s[i] == '(' || s[i] == '{' || s[i] == '[') s1.push(s[i]);
            else if(s1.empty()) return false;  
            else if(s[i] == ')'){
                c = s1.top();
                if(c != '(') return false;
                s1.pop();
            }
            else if(s[i] == '}'){
                c = s1.top();
                if(c != '{') return false;
                s1.pop();
            }
            else if(s[i] == ']'){
                c = s1.top();
                if(c != '[') return false;
                s1.pop();
            }
        }
        if(s1.empty()) return true;
        return false;
    }
};

// Q8 MinStack
// to func as a regular stack and also track the min element and all func at O(1)
// we generate a regular stack with type pair where second holds the min in stack till that ele
class MinStack {
private:
    vector<pair<int,int>> vec;
    int top_num;

public:
    MinStack(){
        top_num = -1;
    }

    void push(int val) {
        int minVal;
        if (top_num == -1) minVal = val;
        else {
            pair<int,int> p = vec[top_num];
            minVal = min(val, p.second);
        }
        vec.push_back({val, minVal});
        top_num++;
    }
    
    void pop() {
        if(top_num >= 0){
        vec.pop_back();
        top_num--;
        }
    }
    
    int top() {
        if(top_num >= 0) return vec[top_num].first;
        else return -1;
    }
    
    int getMin() {
        if(top_num >= 0) return vec[top_num].second;
        else return -1;
    }
};

