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
