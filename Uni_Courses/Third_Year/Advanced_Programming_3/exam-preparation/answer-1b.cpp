#include <iostream>
#include <string>

class Student {
private:
    std::string name;
    int access_count { 0 };

public:
    Student(std::string name) : name {name} {}

    // A: Returns the name
    std::string get_name() { 
        // access_count++; 
        return name; 
    }

    // B: Sets a new name
    void set_name(std::string new_name) { 
        this->name = new_name; 
    }
};

int main() {
    Student std_01 {"Tony Stark"};
    const Student std_02 {"Hulk"};

    // Line 1: Pointer to a constant Student
    Student* ptr { &std_01 };
    ptr->set_name("Thoriana");

    // Line 2: Constant pointer to a Student
    // Student* const cptr { &std_02 };
    // cptr->set_name("Spider Girl");

    // Line 3: Method call
    std::cout << std_02.get_name() << " accessed " << std::endl;

    return 0;
}