#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <stdexcept> //runtime errors

#include "DatasetClass.hpp"

namespace sklearn_cpp {
namespace CSVReader {

    /*
    ==This is an imported function== which we adapted for our Dataset class

    This is only a helper function and not a class so it has to be inline to work properly
    Can be called using the following line:

    sklearn_cpp::CSVReader::read_CSV("path", true/false);

    Parses a CSV file based on the filepath 
    Added a has_header to ignore the first line 
    WORKS ON COMMA SEPARATED CSV!

    */
    inline Dataset read_CSV(const std::string& filename,
                                            bool has_header = true) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        Dataset data;
        std::string line;

        //added functionality to exctract headers
        std::vector<std::string> headers;

        //flag to detect datasets like mnist where the label is stored in the first column called "label"
        bool label_first = false;

        if (has_header && std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) {
                headers.push_back(cell);
            }

            //If the first header is literally called label, we interpret the file as label-first
            if (!headers.empty() && headers[0] == "label") {
                label_first = true;
            }
        }
        data.headers = headers;

        /*
        For every line in the CSV:
        1. Skip if empty or invalid
        2. Extract the features and target
        3. Push the parsed values into the Dataset object

        By default this reader assumes the last column is the target.
        For files like MNIST where the first column is called label, we switch to label-first parsing.
        */
        while (std::getline(file, line)) {
            if (line.empty()) {
                continue;
            }

            std::stringstream ss(line);
            std::string cell;
            std::vector<double> row_values;

            while (std::getline(ss, cell, ',')) {
                row_values.push_back(std::stod(cell));
            }

            if (row_values.size() < 2) {
                throw std::runtime_error("Each row must contain at least 1 feature and 1 target.");
            }

            std::vector<double> features{};
            double target{};

            if (label_first) {
                //label-first format -> first column is y, everything else is X
                target = row_values.front();
                features.assign(row_values.begin() + 1, row_values.end());
            }
            else {
                //default format -> everything but the last column is X, the last column is y
                features.assign(row_values.begin(), row_values.end() - 1);
                target = row_values.back();
            }
            
            //Into X push back the vector of features, and into y push back the parsed target
            data.X.push_back(features);
            data.y.push_back(target);
        }

        //returns Dataset object
        return data;
    }

} //CSVReader
} //sklearn_cpp