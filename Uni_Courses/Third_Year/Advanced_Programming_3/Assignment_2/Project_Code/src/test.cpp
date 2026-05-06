#include "../include/sklearn_cpp/sklearn_cpp.hpp"

#include <iostream>
#include <chrono>

//this file includes all the weights we have already tested so this file can show predict(), among other functionality, without taking 15 minutes to fit the models each time
#include "test_fitted_values.hpp" 

/*

We will use the std::chrono to track elapsed time between start and stop for fitting
Useful to see how long it actually takes to fit the model based on the chosen learning parameters

=============================
ESTIMATED TIME TO TRAIN AND TEST ALL MODELS IS APPROXIMATELY: 16 mins!
But I have a 9800X3D so.....enjoy :)
This is why we included the load() functionality!
=============================

=====================================================================================================================
IF YOU WANT TO LOAD ALL WEIGHTS FOR ALL PARAMETERS FROM OUR PRE-TRAINED FILE, SET THIS TO 1, IF YOU WANT TO RUN .fit() ON ALL MODELS AND MAKE YOUR CPU CRY THEN SET THIS TO 0
=====================================================================================================================
*/
#define LOAD_WEIGHTS_FROM_FILE 1

//we use a template here to overload the out operator to work with the vectors we use internally 
//-> allows us to easily print all the weights in a nice { x, x} format so we can copy paste into our "storage" file
template<typename T> 
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vector) {
    out << "{";

    for (size_t i = 0; i < vector.size(); ++i) {
        //load the value into the out
        out << vector[i]; 

        if (i + 1 < vector.size()) { //only add , when it's not the last element
            out << ", ";
        }
    }

    out << "}";

    return out;
}

int main() {

    //Catch the thrown run-time errors:
    try {

        /*
        ============================
        LINEAR REGRESSION TESTING
        ============================

        TEST 1: CONCRETE DATASET TESTING
        */

        sklearn_cpp::Dataset concrete = sklearn_cpp::CSVReader::read_CSV("data/concrete.csv", true);

        sklearn_cpp::linear_model::LinearRegression lr(1e-7, 50000);   

        if (LOAD_WEIGHTS_FROM_FILE == 1) {
            lr.load(concrete_fitted_weights, concrete_fitted_b);
            std::cout << "Concrete model weights loaded from .hpp" << std::endl;
        }
        else {
            std::cout << "Currently fitting Concrete model..." << std::endl;
            auto start_concrete = std::chrono::steady_clock::now(); //track start time
            lr.fit(concrete); //fit the model

            //track end time and console output it
            auto end_concrete = std::chrono::steady_clock::now(); 
            auto elapsed_ms_concrete = std::chrono::duration_cast<std::chrono::milliseconds>(end_concrete - start_concrete);
            std::cout << "Elapsed time for concrete model training with chosen parameters: \n\t" 
            << elapsed_ms_concrete.count()/1000.0 << " in seconds\n\t"
            << elapsed_ms_concrete.count()/60000.0 << " in minutes\n";
        }

        sklearn_cpp::Dataset concrete_incomplete = sklearn_cpp::CSVReader::read_CSV("data/concrete_incomplete.csv", true);
        sklearn_cpp::Dataset concrete_predictions = lr.predict(concrete_incomplete);
        double r2 = lr.r2_score(concrete_incomplete);

        //concrete_predictions.print(); this print is REALLY big :D
        std::cout << "R2 Score: " << r2 << std::endl;

        //print the weights if the .fit() function was ran
        if(LOAD_WEIGHTS_FROM_FILE == 0) {
            std::cout << "\n== Concrete Linear Regression fitted values== \n";
            std::cout << "std::vector<double> concrete_fitted_weights = "
                    << lr.get_weights()
                    << ";\n";
            std::cout << "double concrete_fitted_b = "
                    << lr.get_b_value()
                    << ";\n";
        }

        std::cout << "\n====================================\n" <<
                      "====================================\n" <<
                      "====================================\n\n";

        /*
        TEST 2: BOSTON HOME PRICE DATASET TESTING
        //DOWNLOADED FROM https://www.geeksforgeeks.org/machine-learning/dataset-for-linear-regression/
        */
        
        sklearn_cpp::Dataset boston = sklearn_cpp::CSVReader::read_CSV("data/Boston.csv", true);

        sklearn_cpp::linear_model::LinearRegression model(4e-7, 
                                                          500000);

        if (LOAD_WEIGHTS_FROM_FILE == 1) {
            model.load(boston_fitted_weights, boston_fitted_b);
            std::cout << "Boston model weights loaded from .hpp" << std::endl;
        }
        else {
            std::cout << "Currently fitting Boston model..." << std::endl;
            auto start_boston = std::chrono::steady_clock::now(); //track start time
            model.fit(boston); //fit the model to the data

            //track end time and console output it
            auto end_boston = std::chrono::steady_clock::now(); 
            auto elapsed_ms_boston = std::chrono::duration_cast<std::chrono::milliseconds>(end_boston - start_boston);
            std::cout << "Elapsed time for Boston model training with chosen parameters: \n\t" 
            << elapsed_ms_boston.count()/1000.0 << " in seconds\n\t"
            << elapsed_ms_boston.count()/60000.0 << " in minutes\n";
        }

        double r2_boston = model.r2_score(boston); //calculate the R2 score on the same dataset
        sklearn_cpp::Dataset predictions = model.predict(boston); //use the predict function on the same dataset and print the predictions
        //predictions.print(); //print dataset this print is REALLY big :D

        std::cout << "R2 Score: " << r2_boston << std::endl; //print R2 score

        //print the weights if the .fit() function was ran
        if (LOAD_WEIGHTS_FROM_FILE == 0) {
            std::cout << "\n== Boston Linear Regression fitted values== n";
            std::cout << "std::vector<double> boston_fitted_weights = "
                    << model.get_weights()
                    << ";\n";
            std::cout << "double boston_fitted_b = "
                    << model.get_b_value()
                    << ";\n";
        }

        std::cout << "\n====================================\n" <<
                      "====================================\n" <<
                      "====================================\n\n";

        /*
        ============================
        LOGISTIC REGRESSION TESTING - BINARY CLASSIFICATION
        ============================

        TEST 1: ECG DATASET TESTING
        */

        sklearn_cpp::Dataset ecg = sklearn_cpp::CSVReader::read_CSV("data/ecg.csv", true);
        sklearn_cpp::linear_model::LogisticRegression logreg(4e-7, //iteration speed
                                                            10000, //number of iterations
                                                            0.01 //lambda
                                                        );

        

        if (LOAD_WEIGHTS_FROM_FILE == 1) {
            logreg.load(ecg_fitted_weights, ecg_fitted_biases, ecg_fitted_labels);
            std::cout << "ECG model weights loaded from .hpp" << std::endl;

        }
        else {
            std::cout << "Currently fitting ECG model..." << std::endl;
            auto start_ecg = std::chrono::steady_clock::now(); //track start time
            logreg.fit(ecg); //fit the model to the data

            //track end time and console output it
            auto end_ecg = std::chrono::steady_clock::now(); 
            auto elapsed_ms_ecg = std::chrono::duration_cast<std::chrono::milliseconds>(end_ecg - start_ecg);
            std::cout << "Elapsed time for model training with chosen parameters: \n\t" 
            << elapsed_ms_ecg.count()/1000.0 << " in seconds\n\t"
            << elapsed_ms_ecg.count()/60000.0 << " in minutes\n";
        }

        
        double accuracy = logreg.accuracy_score(ecg); //calculate the accuracy score on the same dataset
        std::cout << "Accuracy: " << accuracy << std::endl; //print accuracy

        //print the weights if the .fit() function was ran
        if (LOAD_WEIGHTS_FROM_FILE == 0) {
            std::cout << "\n== ECG Logistic Regression fitted values== \n";
            std::cout << "std::vector<std::vector<double>> ecg_fitted_weights = "
                    << logreg.get_weights()
                    << ";\n";
            std::cout << "std::vector<double> ecg_fitted_biases = "
                    << logreg.get_b_values()
                    << ";\n";
            std::cout << "std::vector<int> ecg_fitted_labels = "
                    << logreg.get_labels()
                    << ";\n";
        }

        std::cout << "\n====================================\n" <<
                      "====================================\n" <<
                      "====================================\n\n";
        
        /*
        ============================
        LOGISTIC REGRESSION TESTING - MULTICLASS CLASSIFICATION
        ============================
        */

        //TEST 2: MNIST DATASET TESTING
        //Train on the training file, then evaluate accuracy on the separate test file
        sklearn_cpp::Dataset mnist_train = sklearn_cpp::CSVReader::read_CSV("data/mnist_train.csv", true);
        sklearn_cpp::Dataset mnist_test  = sklearn_cpp::CSVReader::read_CSV("data/mnist_test.csv", true);

        sklearn_cpp::linear_model::LogisticRegression mnist_logreg(1e-5,   //iteration speed
                                                                100,    //number of iterations (small, the list is huge)
                                                                0.0001  //lambda
                                                                );
        

        if (LOAD_WEIGHTS_FROM_FILE == 1) {
            mnist_logreg.load(mnist_fitted_weights, mnist_fitted_biases, mnist_fitted_labels);
            std::cout << "MNIST model weights loaded from .hpp" << std::endl;
        }
        else {   
            
            std::cout << "Currently fitting MNIST Handwritten-Digits model..." << std::endl;
            auto start_mnist = std::chrono::steady_clock::now(); //track start time  

            //Fit only on the training dataset
            mnist_logreg.fit(mnist_train);

            //track end time and console output it
            auto end_mnist = std::chrono::steady_clock::now(); 
            auto elapsed_ms_mnist = std::chrono::duration_cast<std::chrono::milliseconds>(end_mnist - start_mnist);
            std::cout << "Elapsed time for model training with chosen parameters: \n\t" 
            << elapsed_ms_mnist.count()/1000.0 << " in seconds\n\t"
            << elapsed_ms_mnist.count()/60000.0 << " in minutes\n";
        }

        //Check the model on unseen training data and unseen test data separately
        double mnist_train_accuracy = mnist_logreg.accuracy_score(mnist_train);
        double mnist_test_accuracy  = mnist_logreg.accuracy_score(mnist_test);

        std::cout << "MNIST Train Accuracy: " << mnist_train_accuracy << std::endl;
        std::cout << "MNIST Test Accuracy: "  << mnist_test_accuracy  << std::endl;

        //print the weights if the .fit() function was ran
        if (LOAD_WEIGHTS_FROM_FILE == 0) {
            std::cout << "\n== MNIST Logistic Regression fitted values== \n";
            std::cout << "std::vector<std::vector<double>> mnist_fitted_weights = "
                    << mnist_logreg.get_weights()
                    << ";\n";
            std::cout << "std::vector<double> mnist_fitted_biases = "
                    << mnist_logreg.get_b_values()
                    << ";\n";
            std::cout << "std::vector<int> mnist_fitted_labels = "
                    << mnist_logreg.get_labels()
                    << ";\n";
        }

        std::cout << "\n====================================\n" <<
                      "====================================\n" <<
                      "====================================\n\n";

    }                                                     
    catch (const std::exception& e) {
        //print the error in a nice error format onto the console
        std::cerr << "\nCaught exception: " << e.what() << '\n';
    }

    //make sure the console doesn't immediatelly close :D
    std::cout << "\nPress Enter to exit...";
    std::cin.get();

    return 0;
}
