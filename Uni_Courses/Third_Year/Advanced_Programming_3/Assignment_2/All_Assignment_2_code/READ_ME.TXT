TRADEMARK BORKOVEC_CONNOLY_AZAR INC. @2026 
(please contact owners for code licensing)

Daniel Borkovec - 2941124B
Zabiallah Azar – 2893681A
Patrick Connolly – 2777107C

The HPP library is implemented under Project_Code/include, as a proper library should be in includes

The .csv data is under src/data. That file path is necessary for test.cpp to run correctly

All .csv files are read and formatted into our own Dataset class using our own function sklearn_cpp::CSVReader::read_CSV()

The main test file written is under src/demo.cpp
    In that file you'll find the basic rundown of how to run our library:
        including methods to load our pre-trained weights and only test the predict() functionality, or run all test models yourself. 
        For the MNIST handwritten notes, our trained model which you can load was trained on the full MNIST dataset, 
            while the test.cpp automatically trains on a small version of the dataset bundled with the code.

A second file, src/demo_with_graphing uses the matplot library for C++ to graph the results
IN ORDER TO COMPILE THIS DEMO FILE THE MATPLOT++ LIBRARY NEEDS TO BE INSTALLED

ALL FILES INCLUDE COMMENT EXPLANATIONS IN THE TOP