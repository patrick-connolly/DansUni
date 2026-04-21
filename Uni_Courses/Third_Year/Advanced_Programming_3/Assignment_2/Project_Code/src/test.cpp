#include "../include/sklearn_cpp/sklearn_cpp.hpp"

#include <iostream>

#include <chrono>

/*

We will use the std::chrono to track elapsed time between start and stop for fitting
Useful to see how long it actually takes to fit the model based on the chosen learning parameters


=============================
ESTIMATED TIME TO TRAIN AND TEST ALL MODELS IS APPROXIMATELY: 15 mins!
But I have a 9800X3D so.....enjoy :)

*/

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
        
        auto start_concrete = std::chrono::steady_clock::now(); //track start time

        std::cout << "Currently fitting Concrete model..." << std::endl;

        lr.fit(concrete); //fit the model

        //track end time and console output it
        auto end_concrete = std::chrono::steady_clock::now(); 
        auto elapsed_ms_concrete = std::chrono::duration_cast<std::chrono::milliseconds>(end_concrete - start_concrete);
        std::cout << "Elapsed time for concrete model training with chosen parameters: \n\t" 
        << elapsed_ms_concrete.count()/1000.0 << " in seconds\n\t"
        << elapsed_ms_concrete.count()/60000.0 << " in minutes\n";

        sklearn_cpp::Dataset concrete_incomplete = sklearn_cpp::CSVReader::read_CSV("data/concrete_incomplete.csv", true);
        sklearn_cpp::Dataset concrete_predictions = lr.predict(concrete_incomplete);
        double r2 = lr.r2_score(concrete_incomplete);

        //concrete_predictions.print(); this print is REALLY big :D
        std::cout << "R2 Score: " << r2 << std::endl;

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

        std::cout << "Currently fitting Boston model..." << std::endl;

        auto start_boston = std::chrono::steady_clock::now(); //track start time

        model.fit(boston); //fit the model to the data

        //track end time and console output it
        auto end_boston = std::chrono::steady_clock::now(); 
        auto elapsed_ms_boston = std::chrono::duration_cast<std::chrono::milliseconds>(end_boston - start_boston);
        std::cout << "Elapsed time for Boston model training with chosen parameters: \n\t" 
        << elapsed_ms_boston.count()/1000.0 << " in seconds\n\t"
        << elapsed_ms_boston.count()/60000.0 << " in minutes\n";

        double r2_boston = model.r2_score(boston); //calculate the R2 score on the same dataset
        sklearn_cpp::Dataset predictions = model.predict(boston); //use the predict function on the same dataset and print the predictions
        //predictions.print(); //print dataset this print is REALLY big :D

        std::cout << "R2 Score: " << r2_boston << std::endl; //print R2 score

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
        std::cout << "Currently fitting ECG model..." << std::endl;

        auto start_ecg = std::chrono::steady_clock::now(); //track start time

        logreg.fit(ecg); //fit the model to the data

        //track end time and console output it
        auto end_ecg = std::chrono::steady_clock::now(); 
        auto elapsed_ms_ecg = std::chrono::duration_cast<std::chrono::milliseconds>(end_ecg - start_ecg);
        std::cout << "Elapsed time for model training with chosen parameters: \n\t" 
        << elapsed_ms_ecg.count()/1000.0 << " in seconds\n\t"
        << elapsed_ms_ecg.count()/60000.0 << " in minutes\n";

        double accuracy = logreg.accuracy_score(ecg); //calculate the accuracy score on the same dataset
        std::cout << "Accuracy: " << accuracy << std::endl; //print accuracy

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
                                                                100,    //number of iterations (the list is huge)
                                                                0.0001  //lambda
                                                                );
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

        //Check the model on unseen training data and unseen test data separately
        double mnist_train_accuracy = mnist_logreg.accuracy_score(mnist_train);
        double mnist_test_accuracy  = mnist_logreg.accuracy_score(mnist_test);

        std::cout << "MNIST Train Accuracy: " << mnist_train_accuracy << std::endl;
        std::cout << "MNIST Test Accuracy: "  << mnist_test_accuracy  << std::endl;

        std::cout << "\n====================================\n" <<
                      "====================================\n" <<
                      "====================================\n\n";

    }                                                     
    catch (const std::exception& e) {
        std::cerr << "\nCaught exception: " << e.what() << '\n';
    }

    //make sure the console doesn't immediatelly close :D
    std::cout << "\nPress Enter to exit...";
    std::cin.get();

    return 0;
}
