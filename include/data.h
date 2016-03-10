#ifndef DATA_H
#define DATA_H

#include "Sample.h"
#include "DETree.h"

#include <vector>

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
    static constexpr double _stochasticity = 0.0;
    static constexpr double _p = 50000.0;
    static constexpr double _sigma = 1.0;
    static constexpr double _mean = 0.0;
    static constexpr double _minIntensity = 5.0;
    static constexpr double _maxIntensity = 50.0;
    static constexpr double _obsStepSize = (_maxIntensity - _minIntensity) / ((double) (_numberOfDisObs));


    vector<int> listOfStates;
    vector<int> listOfActions;
    vector<double> * low  = new vector<double>();
    vector<double> * high = new vector<double>();
    vector<double> timeChunk;
    DETree observationModel;
    vector<vector<double> > obsSampleList;
    vector<Sample> flatObsList;
    vector<vector<Sample > > obsList;
    vector< vector<vector<double> > > continuousTrajectories;
    vector< vector<vector<int> > > discreteTrajectories;


public:
    Data();
    ~Data();
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
    double getStochasticity(){return _stochasticity;}
    double getP(){return _p;}
    double getSigma(){return _sigma;}
    double getMean(){return _mean;}
    double getMinIntensity(){return _minIntensity;}
    double getMaxIntensity(){return _maxIntensity;}
    double getObsStepSize (){return _obsStepSize;}
    void updateFlatObsList (Sample sample){flatObsList.push_back(sample);}
    void addAContinuousTrajectory(vector<vector<double> > ct){continuousTrajectories.push_back(ct);}
    void addADiscreteTrajectory(vector<vector<int> > dt){discreteTrajectories.push_back(dt);}
    void addObsSample(vector<double> obsSample){obsSampleList.push_back(obsSample);}
    void addObs(vector<Sample > obs){obsList.push_back(obs);}
    void setObsModel(DETree obsModel){observationModel = obsModel;}
};

#endif // DATA_H
