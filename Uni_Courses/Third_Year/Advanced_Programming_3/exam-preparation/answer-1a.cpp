#include <iostream>

// 1. Increments the value of 'n' by 10 using a pointer.
void increment_by_pointer(int* ptr) {
    *ptr += 10;
}

// 2. Doubles the value of 'n' using a reference.
void double_by_reference(int& ref) {
    ref *= 2;
}

// 3. Swaps the address held by two pointers (Pointer to Pointer).
// Note: This should change where the pointers in main point to.
void swap_pointers(int** ptr1, int** ptr2) {
    int* temp = *ptr1; // Store the address pointed to by ptr1
    *ptr1 = *ptr2;     // Make ptr1 point to what ptr2 points to
    *ptr2 = temp;     // Make ptr2 point to what ptr1 originally pointed to
}

int main() {
    int a { 5 };
    int b { 50 };
    
    std::cout << a << '\n';
    std::cout << b << '\n';
    int* pA { &a };
    int* pB { &b };

    // Task 1: Use increment_by_pointer to make 'a' equal to 15.
    increment_by_pointer(pA);
    // Task 2: Use double_by_reference to make 'b' equal to 100.
    double_by_reference(b);
    // Task 3: Use swap_pointers so that pA points to 'b' and pB points to 'a'.
    
    std::cout << "Before swapping pointers:" << std::endl;
    std::cout << "pA points to: " << *pA << std::endl;
    std::cout << "pB points to: " << *pB << std::endl;
    swap_pointers(&pA, &pB);

    std::cout << a;
    std::cout << b;
    std::cout << "After swapping pointers:" << std::endl;
    std::cout << "pA points to: " << *pA << std::endl;
    std::cout << "pB points to: " << *pB << std::endl;
    return 0;
}