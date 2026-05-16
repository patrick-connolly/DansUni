#include <iostream>
#include <string>

class Payment {
public:
    void identify() { 
        std::cout << "Standard Payment" << std::endl; 
    }
    virtual void process() { 
        std::cout << "Processing generic payment..." << std::endl; 
    }
    virtual ~Payment() = default;
};

class CreditCard : public Payment {
public:
    void identify() { 
        std::cout << "Credit Card" << std::endl; 
    }
    void process() override { 
        std::cout << "Processing credit card via secure gateway..." << std::endl; 
    }
};

int main() {
    Payment* payment_ptr = new CreditCard {};
    CreditCard* credit_ptr = new CreditCard {};

    std::cout << "1: "; payment_ptr->identify();
    std::cout << "2: "; payment_ptr->process();
    
    std::cout << "3: "; credit_ptr->identify();
    
    delete payment_ptr;
    delete credit_ptr;

    return 0;
}