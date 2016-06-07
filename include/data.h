#ifndef DATA_H
#define DATA_H

#include "Sample.h"
#include "DETree.h"

#include <vector>
#include <map>
#include <string>

using namespace std;

class Data
{
private:
    static const int _gridSizeX = 40;
    static const int _gridSizeY = 30;
    static const int _stepSize = 10;
    static const int _centerBoxDim = 0;
    static const int _numberOfSamples = 48;
    static const int _classificationStepSize = _numberOfSamples;
    static const int _numberOfDisObs = 5;
    static const int _intensityChangeThreshold = 0;
    static const int _numberOfFeatures = 2;
    static const int _sample_length = 50; // lower than this number (34) will return nan or wrong weights.
    // beacuse lower a number is not suficient to figure out policiy.
    static constexpr double _stochasticity = 0.0;
    static constexpr double _p = 50000.0;
    static constexpr double _sigma = 1.0;
    static constexpr double _mean = 0.0;
    static constexpr double _minIntensity = 5.0;
    static constexpr double _maxIntensity = 50.0;
    static constexpr double _obsStepSize = (_maxIntensity - _minIntensity) / ((double) (_numberOfDisObs));

    string obsModelAddress = "/home/shervin/workspace/qt_ws/Toys/Robust_IRL_TOY/obsModel/obsModel.bin";

    double normalizerFactorForP = 0.0; // for p in sample. uniform

    vector<int> listOfStates;
    vector<int> listOfActions;
    vector<double> *low  = new vector<double>();
    vector<double> *high = new vector<double>();
    vector<double> timeChunk;
    vector<double> weights;
    vector<double> listOfPrT;
    vector<double> listOfTgivenW;
    DETree observationModel;
    vector<vector<double> > obsSampleList;
    vector<Sample> flatObsList;
    vector<vector<Sample > > obsList; // omega
    vector< vector<vector<double> > > continuousTrajectories;
    vector< vector<vector<int> > > discreteTrajectories;
    vector<vector<vector<int>>> allpossibleT;
    map<int,double> stateInitialPriorities; //[State]
    map<int,map<int,double>> policy; //[State][Action] we have to update this in M-step

public:
    Data();
    ~Data();
    double getNormalizerFactorForP(){return normalizerFactorForP;}
    void getNormalizerFactorForP(double n){normalizerFactorForP = n;}
    void updateNormalizerFactorForP(double n){normalizerFactorForP = normalizerFactorForP + n;}
    vector<int> getListOfStates(){return listOfStates;}
    vector<int> getListOfActions(){return listOfActions;}
    vector<Sample> getFlatObsList(){return flatObsList;}
    vector<double> getTimeChunk(){return timeChunk;}
    vector<double> *getLow(){return low;}
    vector<double> *getHigh(){return high;}
    vector<vector<double> > getObsSampleList(){return obsSampleList;}
    vector<vector<Sample > > getObsList(){return obsList;}
    vector< vector<vector<double> > > getListOfContinuousTrajectories(){return continuousTrajectories;}
    vector< vector<vector<int> > > getListOfDiscreteTrajectories(){return discreteTrajectories;}
    vector< vector<vector<int> > > getDiscreteTrajectories() {return discreteTrajectories;}
    int getClassificationStepSize() {return _classificationStepSize;}
    int getGridSizeX(){return _gridSizeX;}
    int getGridSizeY(){return _gridSizeY;}
    int getStepSize(){return _stepSize;}
    int getCenterBoxDim(){return _centerBoxDim;}
    int getNumberOfSamples(){return _numberOfSamples;}
    int getIntensityChangeThreshold(){return _intensityChangeThreshold;}
    int getNumberOfDisObs(){return _numberOfDisObs;}
    int getSampleLength(){return _sample_length;}
    double getStochasticity(){return _stochasticity;}
    string getObsModelAddress(){return obsModelAddress;}
    double getP(){return _p;}
    double getSigma(){return _sigma;}
    double getMean(){return _mean;}
    double getMinIntensity(){return _minIntensity;}
    double getMaxIntensity(){return _maxIntensity;}
    double getObsStepSize(){return _obsStepSize;}
    map<int,double> getListOfStateInitialPriorities(){return stateInitialPriorities;}
    double getStateInitialPriority(int state){return stateInitialPriorities[state];}
    void updateFlatObsList (Sample sample){flatObsList.push_back(sample);}
    void addAContinuousTrajectory(vector<vector<double> > ct){continuousTrajectories.push_back(ct);}
    void addADiscreteTrajectory(vector<vector<int> > dt){discreteTrajectories.push_back(dt);}
    void addObsSample(vector<double> obsSample){obsSampleList.push_back(obsSample);}
    void addObs(vector<Sample > obs){obsList.push_back(obs);}
    void setObsModel(DETree obsModel){observationModel = obsModel;}
    DETree getObsModel(){return observationModel;}
    void updatePolicy(map<int,map<int,double>> p){policy = p;}
    map<int,map<int,double>> getPolicy(){return policy;}
    double getPrActionGivenState(int state,int action){return policy[state][action];}
    vector<vector<vector<int>>> getAllPissibleT(){return allpossibleT;}
    void setAllpossibleT(vector<vector<vector<int>>> apt){allpossibleT = apt;}
    void updatePrT(double prt){listOfPrT.push_back(prt);}
    vector<double> getListOfPrT(){return listOfPrT;}
    void setListOfPrTgivenW(vector<double> lprtw){listOfTgivenW = lprtw;}
    vector<double> getListOfPrTgivenW(){return listOfTgivenW;}
};


#endif // DATA_H
