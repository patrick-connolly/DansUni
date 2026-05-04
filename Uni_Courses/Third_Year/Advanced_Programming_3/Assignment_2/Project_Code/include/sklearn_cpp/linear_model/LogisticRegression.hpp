#pragma once

#include "../DatasetClass.hpp"
#include <cmath>
#include <algorithm> //used for the MODE deciding function and other helpers (used online documentation to implement)

#include <stdexcept> 

/*runtime errors -> used online documentation to implement try{}catch{} methodology to catch runtime errors 
(sometimes the user might enter an incorrect dataset and it throws an error at runtime)

This file only throws the errors, our test.cpp files actually uses try{}catch{} 
*/


/*
=================================
========LOGISTIC REGRESSION=======
=================================

Two "backend" calculation algorithms are implemented into this class, as per the assignment:
Binary classification 
Multiclass classification

=================================
========VECTOR STRUCTURES========
=================================

weights -> [class][feature] -> in binary we only use the first row, in multiclass we use all rows. Hence, for binary the outer size is 1, for multiclass the outer size is K
biases -> [class] -> in binary we only store one bias value, in multiclass its K values
outputs -> [sample][class] -> stores predictions: each sample has a vector of probabilities for each class (decision_function() is the only one which returns raw scores, others use this structure)

====================================
========IMPLEMENTED FEATURES=========
====================================

The fit() function decides which one to use based on the number of unique y values and sets an internal enum
The fit function is the main training function which takes in a Dataset object and trains the model based on the X and y values. 
The gradient descent method is implemented as the main training algoritm.

Prediction API is based on the sklearn API, so we have separated the predict functions into three which may be called separately 
-> like the real library, but the predict() is the main user interface:

1. decision_function() -> returns the raw y prediction values before the sigmoid for BINARY; returns the raw scores for MULTICLASS
2. predict_proba() -> returns predicted probabilities:
   - binary: probability of the class as [sample][0] (because binary only has one positive class (is or is not the class) so only uses first row of the inner)
   - multiclass: full probability vector [sample][class]
3. predict() -> returns the final class predictions based on the probabilities -> MAIN USER INTERFACE
    - binary: we threshold at 0.5 for the probability to make the class distinction
    - multiclass: the choose the class with the highest raw score for the final decision
4. Dataset predict() -> The predict function is overloaded to work with a Dataset object which we generate from the CSV file reader, which is going to be the main way to use our library
5. accuracy_score() -? uses the internal predict() member function and the given loaded Y values to generate the predicted Y values, and outputs an accuracy score in percent
6. load() -> requires an input array for of weights and biases and loads them as the current weights 
          -> Can run the model once, use the getters to output the trained weights and use them to create a trained model instance in another program.

*/

namespace sklearn_cpp {
namespace linear_model {

class LogisticRegression {

    private:
        //These are the learnable parameters -> weights has to be a vector to enable using multi-feature datasets
        std::vector<std::vector<double>> weights; // stores the weights per class, per feature -> [class][feature] (binary is going to use [0][i])
        std::vector<double> biases;               // [class]
        std::vector<int> class_labels; // original labels

        //These are the default learning parameters
        double learning_rate_ = 0.0001;
        size_t n_iterations_ = 2000;
        double lambda = 0.01; //regularization parameter -> used later in fit

        //Track the last loss value from the current fit -> can compare between fits (how low the last loss got)
        double current_loss = 0.0; 

        //enum for deciding whether to implement the binary or multiclass version -> decided based on the fit function
        enum class MODE {
            NOT_FIT,
            BINARY,
            MULTICLASS
        };

        //default to NOT_FIT, will be changed in fit function 
        //essentially acts as a check if a user tries to predict using an unfitted models
        MODE current_mode = MODE::NOT_FIT; 

        /*
        Private helper which counts the number of unique y_values and decides whether to use binary or multiclass:
        Pass by refference to limit copying, internally copy the Y_values only
        ==used online resources to research how to count unique values in a vector===
        */
        MODE decide_mode(const sklearn_cpp::Dataset& data) {
            std::vector<double> y_copy = data.y;
            std::sort(y_copy.begin(), y_copy.end());
            int uniqueCount = std::unique(y_copy.begin(), y_copy.end()) - y_copy.begin();
            if (uniqueCount == 2) {
                return MODE::BINARY;
            } else if (uniqueCount > 2) {
                return MODE::MULTICLASS;
            }
            else {
                //only occurs when the dataset only has one Y value which is...a strange dataset
                throw std::runtime_error("Invalid number of unique classes in y values.");
            }
        }

        /*
        Helper to extract abd store unique class labels for multiclass classification.
        We sort first and return all unique labels
        */
        std::vector<int> extract_labels(const sklearn_cpp::Dataset& data) const {
            std::vector<int> labels;

            /*
            reserves an appropriatelly sized vector (faster for runtime instead of continuously pushing and having to expand the vector)
            Yes the time save here is miniscule so we usually just extend the vector by pushing more values in the later code
            */
            labels.reserve(data.y.size()); 

            for (double y_value : data.y) {
                labels.push_back(int(y_value)); //assume classes are meant to be integers and not double values like 2.1 and 3.8 -> cast to int
            }

            //using std:: library for sorting -> creates a sorted vector
            std::sort(labels.begin(), labels.end());
            //Keep only unique values and erase any left-over repeating labels -> yields a vector of all the possible classes without repeating
            labels.erase(std::unique(labels.begin(), labels.end()), labels.end());

            return labels;
        }

       /*
       Helper which maps an original label value ro the internal class index 0,1,2,...,k-1.
       This is useful because the maths for softmax is much easier with compact class indices.
       Later on we map the predictions back to their original labels
       */
       size_t label_to_index(int label) const {
            for (size_t i = 0; i < class_labels.size(); ++i) {
                if (class_labels[i] == label) {
                    return i;
                }
            }
            throw std::runtime_error("Unknown class label encountered.");
        }

    public:
        //Default constructor for safety -> we have default parameter values anyway
        LogisticRegression() = default;

        //Actual constructor which specifies learning parameters
        LogisticRegression(double learning_rate, size_t n_iterations, double lambda)
            : learning_rate_(learning_rate), n_iterations_(n_iterations), lambda(lambda)
        {
            
        }
    

        /*
        ===================================
        ========FIT IMPLEMENTATION=========
        ===================================
        */

        //This is the main fit functions which takes in the main Dataset object and trains the model
        void fit(const sklearn_cpp::Dataset& data) {

            //Main empty data checks
            if (data.X.empty()) {
                throw std::runtime_error("Dataset is empty.");
            }
            if (data.X.size() != data.y.size()) {
                throw std::runtime_error("X and y size mismatch.");
            }

            //Decide the mode -> binary or multiclass, changes the internal backend of the functions
            this->current_mode = decide_mode(data);

            //Use the helper to store the class labels in the member variable for later use
            this->class_labels = extract_labels(data);

            if(this->current_mode == MODE::BINARY) {
                //these are used in the in the gradient descent calculations and loops -> have to cast to double in order to use in calculations
                const size_t m = data.X.size();
                const size_t num_features = data.X[0].size();

                //In binary, simply use one row of weights and one bias value
                //Uses assign to set the size of the weight vector to match the X values and sets to zero, b value too -> starts a fresh fit
                weights.assign(1, std::vector<double>(num_features, 0.0));
                biases.assign(1, 0.0);

                //This is the main training loop -> runs for n_iterations
                for(size_t iter{0}; iter < n_iterations_; iter++) {

                    //Create the gradient vectors and set to zero for new iteration
                    std::vector<double> dL_dOmega(num_features, 0.0); 
                    double dL_db = 0.0;

                    //1. calculate the y probabilities for the current weights and b -> uses the predict_proba function to get the probabilities based on the sigmoid of the decision function
                    std::vector<std::vector<double>> y_predictions = predict_proba(data.X);

                    //2. Loss function calculation
                    double loss = 0.0;
                    
                    //this is the sigma part for each data point
                    for (size_t i = 0; i < m; i++) {
                        double y_hat = y_predictions[i][0];

                        //Use class[1] -> 1 and class[0] -> 0 internally so it's easier
                        double y_true = (int(data.y[i]) == class_labels[1]) ? 1.0 : 0.0;

                        //loss equation
                        loss += y_true * std::log(y_hat)
                            + (1.0 - y_true) 
                            * std::log(1.0 - y_hat);
                    }

                    //divide by - 1/ m to get the final loss value (no regularization yet)
                    loss = -(1.0 / (double)(m)) * loss;

                    //regularization calculation
                    double reg = 0.0;

                    //range based for loop implementing the sigma
                    for (double w : weights[0]) {
                        reg += w * w; //weight squared
                    }
                    reg += biases[0] * biases[0];

                    /*
                    Final regularized loss -> regularization added at the end
                    */
                    
                    loss += lambda * reg;

                    /*
                    3. Gradient of loss function calculations
                    */
                    for(size_t i{0}; i < m; i++) {
                        //once again protectt against classes not being 1.0 and 0.0
                        double y_true = (int(data.y[i]) == class_labels[1]) ? 1.0 : 0.0;

                        //Calculate the error for the current iteration
                        double error = y_predictions[i][0] - y_true;

                        //Calculate dL_dOmega for each weight
                        for(size_t j{0}; j < data.X[i].size(); j++) {
                            dL_dOmega[j] += (error * data.X[i][j]) + (2 * lambda * weights[0][j]); //add regularization term
                        }
                        dL_db += error + (2 * lambda * biases[0]); //add regularization term
                    }

                    /*
                    Calculate new weights and b based on the gradients and learning rate
                    */
                    for(size_t j{0}; j < weights[0].size(); j++) {
                        weights[0][j] -= (1 / (double)m) * learning_rate_ * dL_dOmega[j];
                    }
                    biases[0] -= (1 / (double)m) * learning_rate_ * dL_db; 
                }

                /*
                //calculate the loss one last time after training loop finished and store it in the member variable
                */
                double loss = 0.0;
                std::vector<std::vector<double>> y_predictions = predict_proba(data.X);
                    
                for (size_t i = 0; i < m; i++) {
                    double y_hat = y_predictions[i][0];

                    /*
                    again map the class values to 1.0 and 0.0 internally
                    (double so that we don't trucate using integer division, even if 1 and 0 is "simpler")
                    */
                    double y_true = (int(data.y[i]) == class_labels[1]) ? 1.0 : 0.0;

                    loss += y_true * std::log(y_hat)
                        + (1.0 - y_true) 
                        * std::log(1.0 - y_hat);                 }

                //divide by - 1/ m to get the final loss value (no regularization yet)
                loss = -(1.0 / (double)(m)) * loss;

                //regularization calculation
                double reg = 0.0;

                //range based for loop implementing the sigma
                for (double w : weights[0]) {
                    reg += w * w; //weight squared
                    }
                reg += biases[0] * biases[0];

                /*
                Final regularized loss -> regularization added at the end
                 */
                    
                loss += lambda * reg;
                this->current_loss = loss; //update the current loss value after training

            }

            //Multiclass implementation
            else {
                //these are used in the gradient descent calculation and loops
                const size_t m = data.X.size();
                const size_t num_features = data.X[0].size();

                //store the original labels and initialise one weight vector and one bias per class
                std::vector<int> multiclass_labels = extract_labels(data);
                const size_t num_classes = multiclass_labels.size();

                //Zero the learnable parameters to retrain, even if the model was trained before
                weights.assign(num_classes, std::vector<double>(num_features, 0.0));
                biases.assign(num_classes, 0.0);

                //This is the main trianing loop -> runs for n_iterations
                for(size_t iter{0}; iter < n_iterations_; iter++) {

                    //Create the gradient containers and set them to zero for the new iteration
                    std::vector<std::vector<double>> dL_dOmega(num_classes, std::vector<double>(num_features, 0.0));
                    std::vector<double> dL_db(num_classes, 0.0);

                    //1. calculate the full probability matrix for the current multiclass weights and biases
                    std::vector<std::vector<double>> y_probabilities = predict_proba(data.X);

                    //2. Loss function calculation
                    double loss = 0.0;

                    //this loop calculates the multiclass cross entropy loss
                    for (size_t i{0}; i < m; i++) {
                        int current_label = static_cast<int>(data.y[i]);
                        size_t true_class_index = label_to_index(current_label); //track the labels using the helper

                        //only the true class contributes to the one-hot cross entropy expression
                        double y_hat_true = y_probabilities[i][true_class_index];

                        loss += -std::log(y_hat_true);
                    }

                    //average over all data points
                    loss = (1.0 / (double)m) * loss;

                    //regularization calculation for all class weights and biases
                    double reg = 0.0;
                    for (size_t k{0}; k < num_classes; k++) {
                        for (double w : weights[k]) {
                            reg += w * w;
                        }
                        reg += biases[k] * biases[k];
                    }

                    /*
                    Final regularized loss -> regularization added at the end
                    */
                    loss += lambda * reg;

                    /*
                    3. Gradient of loss function calculations
                    For each class k, the error term is:
                    p_k - 1 if k is the correct class
                    p_k - 0 otherwise
                    */
                    for(size_t i{0}; i < m; i++) {
                        int current_label = static_cast<int>(data.y[i]);
                        size_t true_class_index = label_to_index(current_label);

                        for (size_t k{0}; k < num_classes; k++) {
                            double target = (k == true_class_index) ? 1.0 : 0.0;
                            double error = y_probabilities[i][k] - target;

                            for(size_t j{0}; j < num_features; j++) {
                                dL_dOmega[k][j] += error * data.X[i][j];
                            }
                            dL_db[k] += error;
                        }
                    }

                    //average the gradients over the dataset size and then add regularization
                    for(size_t k{0}; k < num_classes; k++) {
                        for(size_t j{0}; j < num_features; j++) {
                            dL_dOmega[k][j] = (1.0 / (double)m) * dL_dOmega[k][j] + (2.0 * lambda * weights[k][j]);
                        }
                        dL_db[k] = (1.0 / (double)m) * dL_db[k] + (2.0 * lambda * biases[k]);
                    }

                    /*
                    Calculate new weights and biases based on the gradients and learning rate
                    */
                    for(size_t k{0}; k < num_classes; k++) {
                        for(size_t j{0}; j < num_features; j++) {
                            weights[k][j] -= learning_rate_ * dL_dOmega[k][j];
                        }
                        biases[k] -= learning_rate_ * dL_db[k];
                    }
                }

                /*
                calculate the loss one last time after training loop finished and store it in the member variable
                */
                double loss = 0.0;
                std::vector<std::vector<double>> y_probabilities = predict_proba(data.X);

                for (size_t i{0}; i < m; i++) {
                    int current_label = static_cast<int>(data.y[i]);
                    size_t true_class_index = label_to_index(current_label);

                    double y_hat_true = y_probabilities[i][true_class_index];

                    loss += -std::log(y_hat_true);
                }

                loss = (1.0 / (double)m) * loss;

                double reg = 0.0;
                for (size_t k{0}; k < num_classes; k++) {
                    for (double w : weights[k]) {
                        reg += w * w;
                    }
                    reg += biases[k] * biases[k];
                }

                loss += lambda * reg; //add regulation
                this->current_loss = loss; //store
            }
        }


        /*
        ==========================================
        ======PREDICT VECTOR IMPLEMENTATION=======
        ==========================================
        */

        /*
        THESE FUNCTIONS ARE THE CORE ONES USED BY INTERNALS BUT ALSO CALLABLE DIRECTLY BY THE USER, BUT THE MAIN USER INTERFACE IS predict() WHICH CALLS THESE INTERNALS
        These functions all work with vector inputs and outputs, the main user interface is predict which is the only one overloaded to work with the Dataset class

        
        This function returns the raw scores before the sigmoid
        nested vector because we need one row per multiclass

        the output is structures as [sample][class] -> aka each sample has a vector of probabilities correspoding to each class
        */
        
        std::vector<std::vector<double>> decision_function(const std::vector<std::vector<double>>& X) const {

            if(current_mode == MODE::NOT_FIT) {
                throw std::runtime_error("Model has not been fitted yet.");
            }

            //Nested vector storing the predictins for each sampple, for each class, as per [sample][class]
            std::vector<std::vector<double>> y_prediction_rows{};

            //Binary implementation
            if(current_mode == MODE::BINARY) {

                for(size_t i{0}; i < X.size(); i++) {

                    //We check the current X vector's validity
                    if(X[i].size() != weights[0].size()) {
                        throw std::runtime_error("Input feature size does not match learned weights.");
                    };

                    double current_y_prediction{};

                    //calculate the prediction based on ax + b 
                    for(size_t j{0}; j < X[i].size(); j++) {
                        current_y_prediction += weights[0][j]*X[i][j]; //always the first row for binary
                    }
                    current_y_prediction += biases[0]; //biases only has one entry for binary

                    //store in the predictions first row of the vector
                    y_prediction_rows.push_back({current_y_prediction});

                }

                return y_prediction_rows;
            }

            //Multiclass implementation
            else {

                //loop over all X rows
                for(size_t i{0}; i < X.size(); i++) {

                    //ensure proper sizing
                    if(X[i].size() != weights[0].size()) {
                        throw std::runtime_error("Input feature size does not match learned weights.");
                    }

                    std::vector<double> current_row{};

                    //loop over all X values inside each row
                    for(size_t k{0}; k < weights.size(); k++) {
                        double current_y_prediction{};

                        //calculate the prediction based on ax + b 
                        for(size_t j{0}; j < X[i].size(); j++) {
                            current_y_prediction += weights[k][j] * X[i][j];
                        }

                        //push back the prediction of that row
                        current_y_prediction += biases[k];
                        current_row.push_back(current_y_prediction);
                    }

                    //store all Y prediction rows
                    y_prediction_rows.push_back(current_row);
                }

                return y_prediction_rows;

            }
            

        }

        /*
        This function uses the sigmoid on the predicted y value for BINARY or on the softmax for MULTICLASS
        */
        std::vector<std::vector<double>> predict_proba(const std::vector<std::vector<double>>& X) const {
            if(current_mode == MODE::NOT_FIT) {
                throw std::runtime_error("Model has not been fitted yet.");
            }

            //First we get the raw y predition -> same with linear regression
            std::vector<std::vector<double>> decision_values = decision_function(X);
            std::vector<std::vector<double>> probabilities{};

            if(current_mode == MODE::BINARY) {

                //range based for loop to go through all decision values and apply sigmoid to get the probabilities
                //use pass by reffeence TO MAKE IT FASTER WITHOUT COYPING EVERYTHING
                for (const auto& val : decision_values) {
                    double probability = 1.0 / (1.0 + std::exp(-val[0])); //sigmoid function
                    probabilities.push_back({probability}); 
                }

            }

            //Multiclass implementation
            else {
                for (const auto& row : decision_values) {

                    double largest_score = row[0]; 

                    // Find the largest raw score in this sample using a for loop
                    for (size_t class_index = 1; class_index < row.size(); class_index++) {
                        if (row[class_index] > largest_score) {
                            largest_score = row[class_index];
                        }
                    }

                    //Stores the softmax probabilities for the current sample
                    std::vector<double> current_probabilities(row.size(), 0.0);
                    double denominator = 0.0;

                    // exponentiate each shifted score and sigma the common denominator
                    for (size_t k{0}; k < row.size(); k++) {
                        current_probabilities[k] = std::exp(row[k] - largest_score);
                        denominator += current_probabilities[k];
                    }

                    //Make sure that each possibility is normalize so that their sum is one
                    for (size_t k{0}; k < row.size(); k++) {
                        current_probabilities[k] /= denominator;
                    }

                    probabilities.push_back(current_probabilities);
                }
            }

            return probabilities;
            
        }

        /**
        This function returns the final class predictions based on the probabilities -> MAIN USER INTERFACE
        */
        std::vector<int> predict(const std::vector<std::vector<double>>& X) const {
            if(current_mode == MODE::NOT_FIT) {
                throw std::runtime_error("Model has not been fitted yet.");
            }

            //internal vector to store results
            std::vector<int> class_predictions{};

            //Binary implementation
            if(current_mode == MODE::BINARY) {

                //First we get the probabilities
                std::vector<std::vector<double>> probabilities = predict_proba(X);

                //range based for loop to go through all probabilities and apply threshold of 0.5 to get the class predictions
                //USE PASS BY REFERENCE FOR SPEED
                for (const auto& prob : probabilities) {
                    int class_pred = (prob[0] >= 0.5) ? class_labels[1] : class_labels[0]; //threshold at 0.5 -> also map back to original labels
                    class_predictions.push_back(class_pred);
                }

                return class_predictions;
            }

            //Multiclass implementation
            else {
                //Calculate the raw scores for every class and every sample -> use decision_function and receive the nested vector
                std::vector<std::vector<double>> scores = decision_function(X);

                //Iterate over all the scores and find the highest score
                for (size_t i{0}; i < scores.size(); i++) {
                    size_t best_class_index = 0;
                    double best_score = scores[i][0];

                    //keep track of the highest score and iterate
                    for (size_t k{1}; k < scores[i].size(); k++) {
                        if (scores[i][k] > best_score) {
                            best_score = scores[i][k];
                            best_class_index = k;
                        }
                    }
                    
                    //use the internal member variable class_labels to map back to the original label before returning
                    class_predictions.push_back(class_labels[best_class_index]);
                }

                return class_predictions;
            }

        }

        /*
        ==========================================
        ======PREDICT DATASET IMPLEMENTATION======
        ==========================================
        */

        /*
        This function returns the final class predictions based on the probabilities -> MAIN USER INTERFACE USING OUR DATASET CLASS, which we generate from the CSV files
        This function uses Pass by Value on the data, because we erase the y_value column completely and refill it using the predictions
        -> we don't want to override the input dataset, we output a new one
        This member function is const because it only predicts and outputs a new dataset, but does not change any fitted values in the LogisticRegression object
        */
        sklearn_cpp::Dataset predict(sklearn_cpp::Dataset data) const {
            if(current_mode == MODE::NOT_FIT) {
                throw std::runtime_error("Model has not been fitted yet.");
            }

            //Clear the existing y values to prepare for predictions
            data.y.clear();

            //Binary implementation
            if(current_mode == MODE::BINARY) {

                //First we get the probabilities
                std::vector<std::vector<double>> probabilities = predict_proba(data.X);
            
                //range based for loop to go through all probabilities and apply threshold of 0.5 to get the class predictions
                //again binary only so use the first row only
                for (const auto& prob : probabilities) {
                    double class_pred = (prob[0] >= 0.5) ? (double)class_labels[1] : (double)class_labels[0]; //threshold at 0.5 and map classes back to original header labels
                    data.y.push_back(class_pred);
                }

            }

            //Multiclass implementation
            else {
                std::vector<int> class_predictions = predict(data.X);

                for (int class_pred : class_predictions) {
                    data.y.push_back((double)class_pred);
                }
            }

            return data;
        }

        /*
        ==========================================
        =======ACCURACY SCORE IMPLEMENTATION======
        ==========================================
        */
        double accuracy_score(const sklearn_cpp::Dataset& data) const {
            //once again the basic checks
            if (current_mode == MODE::NOT_FIT) {
                throw std::runtime_error("Model has not been fitted yet.");
            }
            if (data.X.empty()) {
                throw std::runtime_error("Dataset is empty.");
            }
            if (data.X.size() != data.y.size()) {
                throw std::runtime_error("X and y size mismatch.");
            }

            //use the model to predict the Y values based on the X values
            std::vector<int> y_pred = predict(data.X);

            //range based for loop which compares if the predicted class matches the original in the dataset
            //keep track in a correct integer, then use it to calculate the percentage using num_correct/total
            size_t correct = 0;
            for (size_t i = 0; i < data.y.size(); ++i) {
                int y_true = int(data.y[i]);
                if (y_pred[i] == y_true) {
                    ++correct; 
                }
            }

            //Return a percentage accuracy score -> correct predictions / total predictions 
            //Have to cast to double because correct is size_t
            return (double)correct / (double)data.y.size();
        }

        /*
        ===================================
        =========== LOAD FUNCTION =========
        ===================================

        This function takes in a pre-fitted weights set and loads it into the current object -> used so we can fit once and then use the same fitted model in multiple test
        */

        void load(const std::vector<std::vector<double>>& fitted_weights, const std::vector<double>& fitted_b, const std::vector<int>& labels) {
            this->class_labels = labels;
            this->weights = fitted_weights;
            this->biases = fitted_b;

            //Run a quick check based on the number of unique labels to set the mode:
            if (class_labels.size() == 2){
                this->current_mode = MODE::BINARY;
            }
            else {
                this->current_mode = MODE::MULTICLASS;
            }

            
        }

        /*
        ===================================
        ===========GET HELPERS ============
        ===================================
        */
        const std::vector<std::vector<double>>& get_weights() const { 
            return weights; 
        }
        const std::vector<double>& get_b_values() const { 
            return biases; 
        }
        const std::vector<int>& get_labels() const {
            return class_labels;
        }
        double get_current_loss() const {
            return current_loss;
        }

};

} // namespace linear_model
} // namespace sklearn_cpp