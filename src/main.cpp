#include "data.h"
#include "process.h"
#include "RIRL.h"
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
    RIRL rirl;

    //    vector<double> y;
    //    y.push_back(1.0);
    //    y.push_back(0.0);

    //    vector<double> w_initial;
    //    w_initial.push_back(1.0);
    //    w_initial.push_back(1.0);

    //    vector<double> w;
    //    w = rirl.exponentiatedGradient(data,pr,y,w_initial,1.0,0.1);
    //    cout << w.at(0) << endl;
    //    cout << w.at(1) << endl;
    pr.generateTrajectories(data, 10, true);

//    cout << data.getListOfDiscreteTrajectories().at(0).size() << endl;
//    cout << data.getListOfContinuousTrajectories().at(0).size() << endl;
    DETree T = pr.loadObsModel(data);

    DETree O = data.getObsModel();

    Sample a;
    a.values.push_back(0.45357454523266599);
    a.values.push_back(20.3247517246703);
    a.values.push_back(10);
    a.values.push_back(10);
    a.p = 0.010416666666666666;

    double x = T.density_value(a,0.5);

    cout << "x: " << x << endl;

    double xx = O.density_value(a,0.5);

    cout << "xx: " << xx << endl;


    Sample b;
    b.values.push_back(-0.0085455654994017246);
    b.values.push_back(20.15279391134262);
    b.values.push_back(20);
    b.values.push_back(-10);
    b.p = 0.010416666666666666;

    double y = T.density_value(b,0.5);

    cout << "y: " << y << endl;

    double yy = O.density_value(b,0.5);

    cout << "yy: " << yy << endl;

    cout << "Done!" << endl;


//    rirl.constructAllT(data, pr, 1);
//    vector< vector< vector<int>>> x = data.getAllPissibleT();


//    for(int i = 0 ; i < int(x.size()) ; i++){
//        rirl.printNestedVector(x.at(i));
//    }

    //    vector<double> * low  = new vector<double>();
    //    vector<double> * high = new vector<double>();

    //    low->push_back(0);
    //    low->push_back(0);

    //    high->push_back(10);
    //    high->push_back(10);
    //    DETree T(flat, low, high);


    return 0;
}
