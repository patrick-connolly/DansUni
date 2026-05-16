#include <iostream>

// 1. Increments the value of 'n' by 10 using a pointer.
void increment_by_pointer(int* ptr) {
    // Write code here
}

// 2. Doubles the value of 'n' using a reference.
void double_by_reference(int& ref) {
    // Write code here
}

// 3. Swaps the address held by two pointers (Pointer to Pointer).
// Note: This should change where the pointers in main point to.
void swap_pointers(int** ptr1, int** ptr2) {
    // Write code here
}

int main() {
    int a { 5 };
    int b { 50 };
    
    int* pA { &a };
    int* pB { &b };

    // Task 1: Use increment_by_pointer to make 'a' equal to 15.
    // Task 2: Use double_by_reference to make 'b' equal to 100.
    // Task 3: Use swap_pointers so that pA points to 'b' and pB points to 'a'.

    return 0;
}