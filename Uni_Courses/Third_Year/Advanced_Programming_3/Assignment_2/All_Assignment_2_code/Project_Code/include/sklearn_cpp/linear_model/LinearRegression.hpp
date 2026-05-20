#pragma once

/*
==================================
========LINEAR REGRESSION=========
==================================
This is the implementation of the Linear Regression model using gradient descent.
Used online resources for the throw handling features as well as vector functions (like assign and resize)

Implemented features:

1. void fit() member function -> takes in a DataClass object and fits the model using gradient descent. Updates internal weights vector and b value.
2. Predict functions -> two overloads:
    a. std::vector<double> predict(const std::vector<std::vector<double>>& X) -> takes in a vector of X values and returns a vector of predicted Y values based on the current weights and b.
    b. Dataset predict(Dataset data) -> overloaded predict function which takes in a Dataset object and overwrites the internal Y values with the predicted Y values based on the current weights and b. Returns the modified Dataset object.
3. R2 score function -> double r2_score(const Dataset& data) -> takes in a Dataset object and calculates the R2 score based on the current model's predictions and the actual Y values in the dataset.
4. load() -> requires an input array for of weights and biases and loads them as the current weights 
          -> Can run the model once, use the getters to output the trained weights and use them to create a trained model instance in another program.
5. Get helpers -> useful when wanting to export or otherwise use the internal weights, biases and loss

*/

#include "../DatasetClass.hpp"

#include <stdexcept> //runtime errors

namespace sklearn_cpp {
namespace linear_model {

//Little helper to calculate mean
//We can actualy use the templates here to ensure our helper works for int and double, etc.
template<typename T> 
inline double calculate_mean(const std::vector<T> &vect) {
    double mean{0};
    for(size_t i{0}; i < vect.size(); i++) {
        mean += vect[i];
    } 
    return mean / vect.size();
}


class LinearRegression {

private:
    //These are the learnable parameters -> weights has to be a vector to enable using multi-feature datasets
    std::vector<double> weights;
    double b = 0.0;

    //These are the default learning parameters
    double learning_rate_ = 0.001;
    size_t n_iterations_ = 1000;

    //Track the last loss value from the current fit -> can compare between fits
    double current_loss = 0.0; 

public:
    //First default constructor because we will specify a different one too
    LinearRegression() = default;

    //Basic empty-body constructor changing the learning paramters
    LinearRegression(double learning_rate, size_t n_iterations)
        : learning_rate_(learning_rate), n_iterations_(n_iterations) 
    {

    }

    /*
    ===================================
    ========FIT IMPLEMENTATION=========
    ===================================
    */

    //These are the two main exposed functions which this library enables calling
    void fit(const sklearn_cpp::Dataset& data) {

        //Same basic exception handling checks as with predict
        if (data.X.empty()) {
            throw std::runtime_error("Dataset is empty.");
        }
        if (data.X.size() != data.y.size()) {
            throw std::runtime_error("X and y size mismatch.");
        }

        //these are used in the in the gradient descent calculations and loops -> have to cast to double in order to use in calculations
        const size_t m = data.X.size();
        const size_t num_features = data.X[0].size();

        //Uses assign to set the size of the weight vector to match the X values and sets to zero, b value too -> starts a fresh fit
        weights.assign(num_features, 0.0); 
        b = 0.0;

        //track loss outside the loop so we can compare between fits at the end of the training loop
        double loss = 0.0;

        //This is the main training loop -> runs for n_iterations
        for(size_t iter{0}; iter < n_iterations_; iter++) {

            //Create the gradient vectors and set to zero for new iteration
            std::vector<double> dL_dOmega(num_features, 0.0); 
            double dL_db = 0.0;

            //1. Calculate the predictions for the current weights and b -> uses the overloaded predict
            std::vector<double> y_predictions = predict(data.X);

            //2. Reset and calculate the loss function MSE (mean squared error) 
            //(yes its not used in the training loop but good to keep, as we can alter the loop to go off of the loss not number of iterations)
            loss = 0.0;

            //this is the sigma part for each data point
            for(size_t i{0}; i < m; i++) {
                double error = y_predictions[i] - data.y[i];
                loss += error * error;
            }
            loss /= double(m); //divide by m at the end

            /*
            3. Calculate the gradient of loss function calculations
            */

            for(size_t i{0}; i < m; i++) {
                //Calculate the error for the current iteration
                double error = y_predictions[i] - data.y[i];

                //Calculate dL_dOmega for each weight
                for(size_t j{0}; j < data.X[i].size(); j++) {
                    dL_dOmega[j] += (error * data.X[i][j]);
                }
                dL_db += error;
            }

            /*
            4. Calculate new weights and b based on the gradients and learning rate
            */
            for(size_t j{0}; j < weights.size(); j++) {
                weights[j] -= (2 / (double)m) * learning_rate_ * dL_dOmega[j];
            }
            b -= (2 / (double)m) * learning_rate_ * dL_db; 
        }

        //final loss update after training loop
        double latest_loss = 0.0;
        std::vector<double> final_predictions = predict(data.X);         
        for(size_t i{0}; i < m; i++) {
            double error = final_predictions[i] - data.y[i];
            latest_loss += error * error;
        }
        latest_loss /= double(m);
        
        //update the latest loss value
        this->current_loss = latest_loss; 

    }


    /*
    ===================================
    ======PREDICT IMPLEMENTATION=======
    ===================================

    */

    /*
    This predict returns a Y-vector based on an input vector of X values for each 
    */
    
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const {

        //First we check if model has be fitted
        if(weights.empty()) {
                throw std::runtime_error("Model has not been fitted yet.");
            };
        
        //temporary y_value vector
        std::vector<double> y_predictions{};

        for(size_t i{0}; i < X.size(); i++) {

            //We check the current X vector's validity
            if(X[i].size() != weights.size()) {
                throw std::runtime_error("Input feature size does not match learned weights.");
            };

            double current_y_prediction{};

            //calculate the prediction based on ax + b 
            for(size_t j{0}; j < X[i].size(); j++) {
                current_y_prediction += weights[j]*X[i][j];
            }
            current_y_prediction += b;

            //store in the predictions vector
            y_predictions.push_back(current_y_prediction);

        }

        return y_predictions;
    }

    /*
    OVERLOADED predict which works with a Dataset object and outputs a new dataset
    Pass by value because we want to return a new dataset and keep the original unchanged
    */

    sklearn_cpp::Dataset predict(sklearn_cpp::Dataset data) const {

        //First we check if model has be fitted
        if(weights.empty()) {
                throw std::runtime_error("Model has not been fitted yet.");
            };
        
        //Clear Dataset Y_vector just in case to prepare it for the predictions
        data.y.clear();

        for(size_t i{0}; i < data.X.size(); i++) {

            //We check the current X vector's validity
            if(data.X[i].size() != weights.size()) {
                throw std::runtime_error("Input feature size does not match learned weights.");
            };

            double current_y_prediction{};

            //calculate the prediction based on ax + b 
            for(size_t j{0}; j < data.X[i].size(); j++) {
                current_y_prediction += weights[j]*data.X[i][j];
            }
            current_y_prediction += b;

             //store in the data.y predictions vector
            data.y.push_back(current_y_prediction);

        }

        return data;
    }

    /*
    ===================================
    ======R2 SCORE CALCULATION=========
    ===================================
    */
    double r2_score(const sklearn_cpp::Dataset& data) const {
        //Once again necessary checks
        if (weights.empty()) {
            throw std::runtime_error("Model has not been fitted yet.");
        }
        if (data.X.empty()) {
            throw std::runtime_error("Dataset is empty.");
        }
        if (data.X.size() != data.y.size()) {
            throw std::runtime_error("X and y size mismatch.");
        }

        std::vector<double> y_pred = predict(data.X); //use the predict function to get the predicted y values for the given dataset
        const double y_mean = calculate_mean(data.y); //uses helper function for the mean

        //Prepare numerator and demoninator
        double ss_res = 0.0;
        double ss_tot = 0.0;

        for (size_t i = 0; i < data.y.size(); ++i) {
            const double residual = data.y[i] - y_pred[i];
            const double deviation = data.y[i] - y_mean;

            ss_res += residual * residual; //numerator, squared
            ss_tot += deviation * deviation; //denominator, squared
        }

        if (ss_tot == 0.0) {
            throw std::runtime_error("R2 is undefined when all y values are identical.");
        }

        return 1.0 - (ss_res / ss_tot);
    }

    /*
    ===================================
    =========== LOAD FUNCTION =========        
    ===================================

    This function takes in a pre-fitted weights set and loads it into the current object -> used so we can fit once and then use the same fitted model in multiple test
    */

    void load(const std::vector<double>& fitted_weights, double fitted_b) {
        this->weights = fitted_weights;
        this->b = fitted_b;
    }

    /*
    ===================================
    ===========GET HELPERS ============
    ===================================
    */
    const std::vector<double>& get_weights() const { 
        return weights; 
    }
    double get_b_value() const { 
        return b; 
    }
    double get_current_loss() const {
        return current_loss;
    }
};

} // namespace linear_model
} // namespace sklearn_cpp