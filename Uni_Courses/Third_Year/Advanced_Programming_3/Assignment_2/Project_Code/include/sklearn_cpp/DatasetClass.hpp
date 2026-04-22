#pragma once

#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <string>

namespace sklearn_cpp {

class Dataset {
    public: 

        //these are the inputs
        std::vector<std::vector<double>> X;
        std::vector<double> y;

        //possible headers -> empty by default
        std::vector<std::string> headers = {};

        /*
        Useful helpers about the dataset size and the the variables of each X
        */

        size_t num_datapoints() const {
            return X.size();
        };

        size_t datapoint_size() const {
            //if the X isn't empty, then return the number of variables per one X
            return X.size() == 0 ? 0 : X[0].size();
        }

        bool is_empty() const {
            return X.size() == 0;
        };
        
        /*
        =====================
        ==TESTING PURPOSES===
        =====================
        LINEAR DATASET RNG GENERATOR WITH NOISE (ONE X AND ONE Y)
        MOSTLY FOR TESTING PURPOSES!
        Simple contrustor so we can easily create new datasets with different seeds.
        */
         
        Dataset(double k , double b, size_t num_points, int min, int max, int seed, double noise_multiplier, double sigma) 
        {

            std::mt19937 rng(seed); //initialize random using our seed -> used to generate core x values
            std::uniform_real_distribution<double> range(min, max); //clamp random values to a range given in parameters
            std::normal_distribution<double> noise_gauss(0.0, sigma); //the mean for the noise is 0 and the sigma is given in parameters

            for (size_t i{0}; i < num_points; i++) {
                double clean_x = range(rng); //first create a clean x

                // Then add noise to x using the gaussian noise ditribution
                double noisy_x = clean_x + noise_multiplier * noise_gauss(rng);

                // Calculate y given noise x and also add noise
                double noisy_y = k * noisy_x + b + noise_multiplier * noise_gauss(rng); // calculate y + add noise

                //finally push the noisy x, full row, and y to the dataset
                X.push_back({noisy_x}); 
                y.push_back(noisy_y);
            }
        };

        //default contructor (empty) - just in case
        Dataset() = default;

        //useful print function -> also prints headers if its not empty (which it is by default)
        void print() const {  
            //if there are headers, print them
            if(headers.size() != 0) {
                std::cout << "Headers: ";
                for (size_t j{0}; j < headers.size(); j++) {
                    std::cout << headers[j];
                    if (j + 1 < headers.size()) {
                        std::cout << ", ";
                    }
                }
                std::cout << "\n";
            }

            //print the actual values
            for (size_t i{0}; i < X.size(); i++) {

                std::cout << "| Row " << i << ": X = ";

                for (size_t j{0}; j < X[i].size(); j++) {
                    std::cout << X[i][j];
                    if (j + 1 < X[i].size()) {
                        std::cout << ", "; //this check avoids the last trailing comma, purely aesthetic
                    }
                }

                std::cout << "| Y = " << y[i] << " \n";
            }

            std::cout << "\n";
        };
};

} // namespace sklearn_cpp