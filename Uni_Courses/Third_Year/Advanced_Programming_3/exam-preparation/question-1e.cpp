#include <iostream>
#include <vector>
#include <string>

struct Student {
    std::string name;
    double grade;
};

int main() {
    std::vector<Student> class_list {
        {"Alice", 75.0},
        {"Bob", 42.5},
        {"Charlie", 88.0}
    };

    // Task 1: Fix this loop
    // Logic: If a grade is below 50, "curve" it by adding 10 points.
    for (Student s : class_list) {
        if (s.grade < 50.0) {
            s.grade += 10.0;
        }
    }

    // Task 2: Fix this loop
    // Logic: Print all student names and grades.
    std::cout << "Class List:" << '\n';
    for (Student s : class_list) {
        std::cout << s.name << ": " << s.grade << '\n';
    }

    return 0;
}