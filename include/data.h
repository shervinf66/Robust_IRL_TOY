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
    static const int _numberOfRawStates = 5;//21; //new // it should be an odd number
    static const int _numberOfActions = 2; //new
    static const int _gridSizeX = 20; //shows the size of each cell //changed
    static const int _gridSizeY = 20; //shows the size of each cell //changed
    static const int _stepSize = 10;
    static const int _max_iter = 1000;//new maximum iteration for qValueSoftMaxSolver at Process
    //    static const int _centerBoxDim = 0;
    static const int _numberOfSamples = 48;
    static const int _classificationStepSize = _numberOfSamples;
    static const int _numberOfDisObs = 5;
    //    static const int _intensityChangeThreshold = 0;
    static const int _numberOfFeatures = ((_numberOfRawStates + 1) / 2) + 1 + 1; //the last one is for the missing features
                                                                                 // and it will be located at one before last in the feature vector
    int sample_length;// = 1000; // lenght of T
    static constexpr double _stochasticity = 0.0;
    static constexpr double _p = 1000000.0; // low numbers very noisy 50000.0;
    static constexpr double _sigma = 1;
    static constexpr double _gamma = 0.99;
    static constexpr double _mean = 0.0;
    //    get ride of them if no problem
    static constexpr double _minIntensity = 5.0;
    static constexpr double _maxIntensity = 50.0;
    static constexpr double _obsStepSize = (_maxIntensity - _minIntensity) / ((double) (_numberOfDisObs));

    string obsModelAddress = "/home/shervin/workspace/qt_ws/Toys/Robust_IRL_TOY/obsModel/obsModel.bin";

    double normalizerFactorForP = 0.0; // for p in sample. uniform

    vector<int> listOfRawStates; //changed
    vector<vector<int>> listOfStates; //new <rawstate, orienaation> ori==> 0 = east, 1 = west
    vector<int> listOfActions;
    vector<double> *low  = new vector<double>();
    vector<double> *high = new vector<double>();
    vector<double> timeChunk;
    vector<double> learnerWeights;
    vector<double> expertWeights;
    vector<double> listOfPrT;
    vector<double> listOfTgivenW;
    DETree observationModel;
    vector<vector<double> > obsSampleList;
    vector<Sample> flatObsList;
    vector<Sample> deterministicObsModel;
    // dont need this any more just use combinae all observation and then use flatobslist
    vector<vector<Sample > > obsList; // omega
    vector< vector<vector<double> > > continuousTrajectories;
    vector< vector<vector<int> > > discreteTrajectories;
    vector<vector<vector<int>>> allpossibleT;
    map<vector<int>,double> stateInitialPriorities; //[State] // modified for debug
    map<vector<int>,map<int,double>> LearnerPolicy; //[State][Action] we have to update this in M-step
    map<vector<int>,map<int,double>> ExpertPolicy; //[State][Action] we have to update this in M-step

public:
    Data();
    ~Data();
    double getNormalizerFactorForP(){return normalizerFactorForP;}
    double getGamma(){return _gamma;}
    void setNormalizerFactorForP(double n){normalizerFactorForP = 1 / n;}
    void updateNormalizerFactorForP(){normalizerFactorForP = normalizerFactorForP + 1;} //changed
    vector<int> getListOfRawStates(){return listOfRawStates;}
    vector<vector<int>> getListOfStates(){return listOfStates;} //new
    vector<int> getListOfActions(){return listOfActions;}
    vector<Sample> getFlatObsList(){return flatObsList;}
    vector<Sample> getDeterministicObsModel(){return deterministicObsModel;}
    void setDeterministicObsModel(vector<Sample> model){deterministicObsModel = model;}
    vector<double> getTimeChunk(){return timeChunk;}
    vector<double> getExpertWeights(){return expertWeights;} //new
    vector<double> *getLow(){return low;}
    vector<double> *getHigh(){return high;}
    vector<vector<double> > getObsSampleList(){return obsSampleList;}
    vector<vector<Sample > > getObsList(){return obsList;}
    vector< vector<vector<double> > > getListOfContinuousTrajectories(){return continuousTrajectories;}
    vector< vector<vector<int> > > getListOfDiscreteTrajectories(){return discreteTrajectories;}
    vector< vector<vector<int> > > getDiscreteTrajectories() {return discreteTrajectories;}
    int getNumberOfRawStates() {return _numberOfRawStates;} //new
    int getNumberOfAction() {return _numberOfActions;} //new
    int getNumberOfFeatures() {return _numberOfFeatures;} //new
    int getClassificationStepSize() {return _classificationStepSize;}
    int getGridSizeX(){return _gridSizeX;}
    int getGridSizeY(){return _gridSizeY;}
    int getStepSize(){return _stepSize;}
    int getMaxIter(){return _max_iter;}
    //    int getCenterBoxDim(){return _centerBoxDim;}
    int getNumberOfSamples(){return _numberOfSamples;}
    //    int getIntensityChangeThreshold(){return _intensityChangeThreshold;}
    int getNumberOfDisObs(){return _numberOfDisObs;}
    void setSampleLength(int sl){ sample_length = sl;}
    int getSampleLength(){return sample_length;}
    double getStochasticity(){return _stochasticity;}
    string getObsModelAddress(){return obsModelAddress;}
    double getP(){return _p;}
    double getSigma(){return _sigma;}
    double getMean(){return _mean;}
    double getMinIntensity(){return _minIntensity;}
    double getMaxIntensity(){return _maxIntensity;}
    double getObsStepSize(){return _obsStepSize;}
    map<vector<int>,double> getListOfStateInitialPriorities(){return stateInitialPriorities;}
    double getStateInitialPriority(vector<int> state){return stateInitialPriorities[state];} // modified for debug
    void updateFlatObsList (Sample sample){flatObsList.push_back(sample);}
    void addAContinuousTrajectory(vector<vector<double> > ct){continuousTrajectories.push_back(ct);}
    void addADiscreteTrajectory(vector<vector<int> > dt){discreteTrajectories.push_back(dt);}
    void addObsSample(vector<double> obsSample){obsSampleList.push_back(obsSample);}
    void addObs(vector<Sample > obs){obsList.push_back(obs);}
    void setObsModel(DETree obsModel){observationModel = obsModel;}
    DETree getObsModel(){return observationModel;}
    void updateLearnerPolicy(map<vector<int>,map<int,double>> p){LearnerPolicy = p;}
    map<vector<int>,map<int,double>> getLearnerPolicy(){return LearnerPolicy;}
    void setExpertPolicy(map<vector<int>,map<int,double>> p){ExpertPolicy = p;}
    map<vector<int>,map<int,double>> getExpertPolicy(){return ExpertPolicy;}
    double getPrActionGivenState(vector<int> state,int action){return LearnerPolicy[state][action];}
    vector<vector<vector<int>>> getAllPissibleT(){return allpossibleT;}
    void setAllpossibleT(vector<vector<vector<int>>> apt){allpossibleT = apt;}
    void updatePrT(double prt){listOfPrT.push_back(prt);}
    vector<double> getListOfPrT(){return listOfPrT;}
    void setListOfPrTgivenW(vector<double> lprtw){listOfTgivenW = lprtw;}
    vector<double> getListOfPrTgivenW(){return listOfTgivenW;}
};


#endif // DATA_H
