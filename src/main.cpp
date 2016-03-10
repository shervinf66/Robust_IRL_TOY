#include "data.h"
#include "process.h"
#include "DETree.h"

#include <cstdlib>
#include <iostream>
#include <fstream>



using namespace std;

int main()
{
    srand(1366); // For reproducable results

    Data data = Data();
    Process pr;

    pr.generateTrajectories(data, 1, true);

    cout << data.getListOfDiscreteTrajectories().at(0).size() << endl;
    cout << data.getListOfContinuousTrajectories().at(0).size() << endl;

//    vector<double> * low  = new vector<double>();
//    vector<double> * high = new vector<double>();

//    low->push_back(0);
//    low->push_back(0);

//    high->push_back(10);
//    high->push_back(10);
//    DETree T(flat, low, high);


    return 0;
}
