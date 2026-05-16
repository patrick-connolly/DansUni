#include <iostream>
#include <string>

// --- Write your Function Template here ---


// -----------------------------------------

struct Box {
    int volume;
    // Overloading the > operator for Box comparison
    bool operator>(const Box& other) const {
        return volume > other.volume;
    }
};

int main() {
    // Test Case 1: Integers
    int a { 10 }, b { 20 };
    std::cout << "Max int: " << find_max(a, b) << '\n';

    // Test Case 2: Strings
    std::string str_01 { "Apple" };
    std::string str_02 { "Zebra" };
    std::cout << "Max string: " << find_max(str_01, str_02) << '\n';

    // Test Case 3: Custom Objects
    Box box_01 { 500 }, box_02 { 1000 };
    Box biggest = find_max(box_01, box_02);
    std::cout << "Max Box volume: " << biggest.volume << '\n';

    return 0;
}