#include <iostream>
#include <string>

// --- Write your Function Template here ---
template <typename T, typename S>
auto find_max(const T& a, const S& b) {
    return (a < b) ? b : a;
}

// -----------------------------------------

struct Box {
    int volume;
    // Overloading the > operator for Box comparison
    // bool operator>(const Box& other) const {
    //     return volume > other.volume;
    // }

    bool operator<(const Box& other) const {
        return volume < other.volume;
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

    std::cout << find_max(10, 20.0) << '\n'; // Test Case 4: Mixed Types (int and double)
    return 0;
}