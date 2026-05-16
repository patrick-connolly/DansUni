#include <iostream>
#include <cmath>

class ComplexNumber {
private:
    double re;
    double im;

public:
    // Constructor with list initialization and default values
    // ComplexNumber(double re = 0, double im = 0) : re{re}, im {im} {}

    // Overloading the assignment operator
    // ComplexNumber& operator=(const ComplexNumber& other) {
    //     if (this != &other) {
    //         this->re = other.re;
    //         this->im = other.im;
    //     }
    //     return *this;
    // }       // Use this as a = b

    // Overloading the + operator
    // ComplexNumber operator+(const ComplexNumber& other) const {
    //     return ComplexNumber(re + other.re, im + other.im);
    // }       // Use this as c = a + b

    // // Overloading the * operator
    // ComplexNumber operator*(const ComplexNumber& other) const {
    //     // double new_re = (re * other.re) - (im * other.im);
    //     // double new_im = (re * other.im) + (im * other.re);
    //     return ComplexNumber((re * other.re) - (im * other.im), (re * other.im) + (im * other.re));
    // }

    // // Power function z^n
    // ComplexNumber pow(const int n) const {
    //     ComplexNumber result(1, 0);     // z^0 = 1
    //     for (int i { 0 }; i < n; ++i) {
    //         result = result * (*this);
    //     }
    //     return result;
    // }

    // // Friend function to overload << for ostream
    // friend std::ostream& operator<<(std::ostream& os, const ComplexNumber& cn) {
    //     if (cn.re != 0 && cn.im > 0) {
    //         os << cn.re << " + " << cn.im << "i";
    //     } else if (cn.re != 0 && cn.im < 0) {
    //         os << cn.re << " - " << std::abs(cn.im) << "i";
    //     } else if (cn.re == 0 && cn.im != 0) {
    //         os << cn.im << "i";
    //     } else {
    //         os << cn.re;
    //     }
    //     return os;          // Think about this: std::cout << ComplexNumber {0, 2} << std::endl;
    // }   // use this as std::cout << a << b << c << std::end;

    virtual void draw() = 0;
};

// Derived class RealNumber
class RealNumber : public ComplexNumber {

public:
    // RealNumber(double value) : ComplexNumber(value, 0) {}

    // void draw() { 
    //     std:: cout << "Draw the real number on the real axis." << std::endl; 
    // }
};

int main() {
    // // Dynamic allocation of an array of 5 ComplexNumber objects
    // ComplexNumber* z_array = new ComplexNumber[5];

    // // Initialize using for-loop
    // // Pattern: z_j = j + (j+1) \sqrt{-1}
    // for (int j = 0; j < 5; ++j) {
    //     z_array[j] = ComplexNumber(static_cast<double>(j + 1), 
    //                                static_cast<double>(j + 2));
    // }

    // // Compute sum s = \sum\limits_{j} z_j
    // ComplexNumber sum { 0, 0 };
    // for (int j = 0; j < 5; ++j) {
    //     sum = sum + z_array[j];
    // }
    // std::cout << "sum = " << sum << std::endl;

    // // Compute Lambda = z^0 + z^1 + z^2 + z^3 + z^4 + z^5 for z = 1 + 2i
    // ComplexNumber z {1, 2};
    // ComplexNumber lambda {0, 0};
    // for (int n = 0; n <= 5; ++n) {
    //     lambda = lambda + z.pow(n);
    // }
    // std::cout << "Lambda = " << lambda << std::endl;

    // // Clean up dynamic memory
    // delete[] z_array;

    return 0;
}