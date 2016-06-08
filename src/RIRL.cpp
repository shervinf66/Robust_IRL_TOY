#include "data.h"
#include "process.h"
#include "RIRL.h"

#include <vector>
#include <map>
#include <algorithm>
#include <float.h>

RIRL::RIRL()
{

}

RIRL::~RIRL()
{

}

vector<double> RIRL::calcFeatureExpectationLeft(Data &data, Process &pr, vector<double> weights){
    //bakward pass
    map<int,double> z_s; //[State]
    map<int,map<int,double>> z_a; //[State][Action]
    vector<int> listOfStates = data.getListOfStates();
    vector<int> listOfActions = data.getListOfActions();

    for (int i = 0 ; i < int(listOfStates.size()) ; i++){
        if (pr.isTerminalState(listOfStates.at(i))){
            z_s[listOfStates.at(i)] = 1.0;
        }else{
            z_s[listOfStates.at(i)] = 0.0;
        }
    }

    int sampleLength = data.getSampleLength();
    for (int i = 0 ; i < sampleLength ; i++){
        // calculating z_ai,j
        for (int j = 0 ; j < int(listOfStates.size()) ; j++){ //j ==> state
            int currentState = listOfStates.at(j);
            if (pr.isTerminalState(currentState)){
                vector<double> features = pr.getFeatures(currentState,0); //since we are in terminal state the action is 0 = no op.
                double reward = exp(pr.calcInnerProduct(features,weights));
                z_a[currentState][0] = reward * z_s[currentState];
            }else {
                for (int k = 0 ; k < int(listOfActions.size()) ; k++){ //k ==> acttion
                    int action = listOfActions.at(k);
                    double sum = 0.0;
                    for (int l = 0 ; l < int(listOfStates.size()) ; l++){ //l ==> next state
                        int nextState = listOfStates.at(l);
                        vector<double> features = pr.getFeatures(currentState,action); // feature of the current state and action
                        double reward = exp(pr.calcInnerProduct(features,weights));
                        double probabilityOfTheNextState; // here we have to calc Pr(s_k|s_i,a_i,j)
                        probabilityOfTheNextState =
                                pr.probablityOfNextStateGivenCurrentStateAction(data,nextState, currentState, action);
                        sum += probabilityOfTheNextState * reward * z_s[nextState];
                    }
                    z_a[currentState][action] = sum;
                }
            }
        }
        // calculating z_si
        for (int j = 0 ; j < int(listOfStates.size()) ; j++){ //j ==> state foreach(i; model.S()) {
            int currentState = listOfStates.at(j);
            double sum = 0.0;
            if (pr.isTerminalState(currentState)) {
                sum = sum + z_a[currentState][0];
            } else {
                for (int k = 0 ; k < int(listOfActions.size()) ; k++){ //k ==> acttion
                    int action = listOfActions.at(k);
                    sum = sum + z_a[currentState][action];
                }
            }
            z_s[currentState] = sum + 1;
        }
    }
    // local action probability computation
    map<int,map<int,double>> policy; //[State][Action]
    for (int i = 0 ; i < int(listOfStates.size()) ; i++){ //i ==> state
        int state = listOfStates.at(i);
        if (pr.isTerminalState(state)) {
            policy[state][0] = z_a[state][0] / z_s[state];
        } else {
            for (int j = 0 ; j < int(listOfActions.size()) ; j++){ //j ==> acttion
                int action = listOfActions.at(j);
                policy[state][action] = z_a[state][action] / z_s[state];
            }
        }
    }
    data.updatePolicy(policy);
    //forward pass
    //initialize D_si,0
    vector< map < int, double> > d;// this a vector of maps that contains all D_sk for t=0 to t=N
    map < int, double> initials = data.getListOfStateInitialPriorities(); // initiall probability of the states
    d.push_back(initials);
    //recursive part
    for (int t = 1 ; t < sampleLength ; t++) {
        map < int, double> temp;
        for (int k = 0 ; k < int(listOfStates.size()) ; k++) {
            int state_k = listOfStates.at(k);
            temp[state_k] = 0.0;
            for (int i = 0 ; i < int(listOfStates.size()) ; i++) {
                int state_i = listOfStates.at(i);
                if (pr.isTerminalState(state_i)) {
                    if (state_i == state_k)
                        temp[state_k] += d.at(t-1)[state_k] * policy[state_k][0];
                } else {
                    for (int j = 0 ; j < int(listOfActions.size()) ; j++) {
                        int action = listOfActions.at(j);
                        double probabilityOfTheNextState; // here we have to calc Pr(s_k|s_i,a_i,j)
                        probabilityOfTheNextState =
                                pr.probablityOfNextStateGivenCurrentStateAction(data,state_k, state_i, action);
                        temp[state_k] += d.at(t-1)[state_i] * policy[state_i][action] *  probabilityOfTheNextState;
                    }
                }
            }
        }
        d.push_back(temp);
    }
    // here to calculate feature excptation but first calculate DS
    vector < double> featureExpectation(weights.size(), 0.0); //double [] returnval = new double[ff.dim()]; returnval[] = 0;

    // calculate D_s (6)
    map<int, double> ds; //D_s[State] ==> stateVisitationFrequency
    for (int t = 1 ; t < sampleLength ; t++) {
        for (int i = 0 ; i < int(listOfStates.size()) ; i++) {
            int state_i = listOfStates.at(i);
            if (t == 0) {
                ds[i] = 0.0;
            }
            ds[state_i] += d.at(t)[state_i];
        }
    }

    for (int i = 0 ; i < int(listOfStates.size()) ; i++) {
        int state_i = listOfStates.at(i);
        if (pr.isTerminalState(state_i)) {
            vector <double> featureVector = pr.getFeatures(state_i,0);
            featureExpectation = pr.add(featureExpectation ,pr.multiply(ds.at(state_i) * policy[state_i][0] , featureVector));
        } else {
            for (int j = 0 ; j < int(listOfActions.size()) ; j++) {
                int action = listOfActions.at(j);
                vector <double> featureVector = pr.getFeatures(state_i,0);
                featureExpectation = pr.add(featureExpectation ,pr.multiply(ds[state_i] * policy[state_i][action] , featureVector));
            }
        }
    }
    return featureExpectation;
}

vector<double> RIRL::exponentiatedGradient(Data &data, Process &pr, vector<double> y,//y is coming from e-step. it is the feature expectation
                                           vector<double> w, double c, double err){
    double y_norm = pr.l1norm(y);
    if (y_norm != 0){
        y = pr.divide(y_norm, y);
    }

    for (int i = 0 ; i < int(w.size()) ; i++) {
        w.at(i) = abs(w.at(i));
    }
    double w_norm = pr.l1norm(w);

    if (w_norm != 0){
        w = pr.divide(w_norm, w);
    }

    double diff;
    double lastdiff = DBL_MAX;

    do {

        vector<double> y_prime = calcFeatureExpectationLeft(data,pr,w);

        double y_prime_norm = pr.l1norm(y_prime);
        if (y_prime_norm != 0)
            y_prime = pr.divide(y_prime_norm, y_prime);

        vector<double> next_w;
        for (int i = 0 ; i < int(w.size()) ; i++) {
            next_w.push_back(w.at(i) * exp(-2*c*(y_prime.at(i) - y.at(i)))) ;
        }

        double norm = pr.l1norm(next_w);
        if (norm != 0)
            next_w = pr.divide(norm, next_w);

        vector<double>  test = w;
        //        test[] -= next_w[];
        for (int i = 0 ; i < int(test.size()) ; i++){
            test.at(i) = test.at(i) - next_w.at(i);
        }
        diff = pr.l2norm(test);

        w = next_w;
        c /= 1.05;
        /*			if (diff > lastdiff) {
            c /= 1.1;
        } else {
            c *= 1.05;
        }*/
        lastdiff = diff;

    } while (diff > err);

    return w;
}

vector<Node> RIRL::returnChildren(Data &data, Process &pr, Node node){
    vector<int> listOfStates = data.getListOfStates();
    vector<int> listOfActions = data.getListOfActions();
    vector<Node> result;

    if(node.isFakeNode){
        // generate a list of initial states
        for(int i = 0 ; i < 1 ; i++){//just consider 10 for initial state
            for(int j = 0 ; j < int(listOfActions.size()) ; j++){
                Node child;
                child.isFakeNode = false;
                child.isInitalState = true;
                child.state = listOfStates.at(i);
                child.action = listOfActions.at(j);
                result.push_back(child);
            }
        }
    }else{
        for(int i = 0 ; i < int(listOfStates.size()) ; i++){
            for(int j = 0 ; j < int(listOfActions.size()) ; j++){
                Node child;
                child.previousState = node.state;
                child.previousAction = node.action;
                child.isFakeNode = false;
                child.isInitalState = false;
                if(!pr.areAdj(listOfStates.at(i), node.state)
                        || pr.isBlockedlState(listOfStates.at(i))){//check for adjacent states
                    break;
                }else if(pr.isTerminalState(listOfStates.at(i))){
                    child.state = listOfStates.at(i);
                    child.action = 0;//do nothing in the terminal state
                    result.push_back(child);
                    break;
                }else{
                    child.state = listOfStates.at(i);
                    child.action = listOfActions.at(j);
                    result.push_back(child);
                }
            }
        }
    }
    return result;
}

vector<double> RIRL::eStep(Data &data, Process &pr, vector<vector<Sample>> allW){ // E-step main function
    // to retain the ultimate feature count
    vector<double> f(2,0.0);

    // go through all observation sequence
    for(int i = 0 ; i < int(allW.size()) ; i++){
        vector<Sample> w = allW.at(i);
        double normalizerVectorForPrTgivenW = 0.0;
        vector<double> featureExpectation(2,0.0);
        vector<double> featureVector(2,0.0);
        Node node;
        node.isFakeNode = true;
        eStepRecursiveUtil(data,pr,w,node,0,1,1,0,
                           normalizerVectorForPrTgivenW,
                           featureVector,featureExpectation);
        //normlize featureExpectation
        for (int j = 0 ; j < int(featureExpectation.size()) ; j++){
            featureExpectation.at(j) = featureExpectation.at(j) / normalizerVectorForPrTgivenW;
        }
        f = pr.add(f,featureExpectation);
    }
    return f;
}

void RIRL::eStepRecursiveUtil(Data &data, Process &pr, vector<Sample> w, Node node, int level
                              ,double prT, double prWgivenT,double normalizerForObsModel,
                              double &normalizerVectorForPrTgivenW, vector<double> &featureVector,
                              vector<double> &featureExpectationVector){ // E-step recursive
    //    cout << level << endl;
    if(level == int(w.size())){
        double prWgivenTNormalized = prWgivenT / normalizerForObsModel;
        double prTgivenW = prWgivenTNormalized * prT;
        // accumulate normalizer after compilation of each T
        normalizerVectorForPrTgivenW = normalizerVectorForPrTgivenW + prTgivenW;
        //later i need to normalize normalize this featureExpectationVector in the eStep function
        //        printVector(featureVector); // for ken
        if(featureVector.at(0)> 1 || featureVector.at(1)> 1){
            if(node.state == 33){
                cout<< "33!" << endl;
                printVector(featureVector);
            }else if(node.state == 23){
                cout<< "23!" << endl;
                printVector(featureVector);
            }
        }
        featureExpectationVector = pr.add(featureExpectationVector,
                                          pr.multiply(prTgivenW, featureVector));
        return;
    }
    vector<Node> children = returnChildren(data, pr, node);
    for(int i = 0 ; i < int(children.size()) ; i++){

        Node child = children.at(i);
        vector<double> tempFeatureVector = featureVector;

        //updating common parameters before recursive call.
        int newLevel = level + 1;
        double newPrT;
        Sample s = w.at(level);
        s.values.push_back(child.state);
        s.values.push_back(child.action);
        double newPrWgivenT = prWgivenT * data.getObsModel().density_value(s,0.5);
        double newNormalizerForObsModel = normalizerForObsModel + newPrWgivenT;
        featureVector = pr.add(featureVector,pr.getFeatures(child.state,child.action));

        //check if in the initial state so the updating parameters will be different
        if(child.isInitalState){// updating parameters before recursive call. if we are in initial state.
            newPrT = prT * data.getStateInitialPriority(child.state);

        }else{ // updating parameters before recursive call. if we are not in initial state.
            newPrT = prT *
                    pr.probablityOfNextStateGivenCurrentStateAction(
                        data,child.state,
                        child.previousState,child.previousAction) *
                    data.getPolicy()[child.state][child.action];
        }
        // recursive call
        eStepRecursiveUtil(data,pr,w,child,newLevel,newPrT,newPrWgivenT,newNormalizerForObsModel,
                           normalizerVectorForPrTgivenW,featureVector,featureExpectationVector);
        // restore feature vector for going back to the previous level
        // normalizerVectorForPrTgivenW and featureExpectationVector must keep acumulate for all Ts
        featureVector = tempFeatureVector;
    }
}

void RIRL::initializePolicy(Data & data, Process &pr){
    vector<int> listOfStates = data.getListOfStates();
    vector<int> listOfActions = data.getListOfActions();
    map<int,map<int,double>> policy; //[State][Action]
    // initialize policy so the Pr(T) would not be zero at begining!
    for (int i = 0 ; i < int(listOfStates.size()) ; i++){ //i ==> state
        int state = listOfStates.at(i);
        if (pr.isTerminalState(state)) {
            policy[state][0] = 1.0; // uniform
        } else {
            for (int j = 0 ; j < int(listOfActions.size()) ; j++){ //j ==> acttion
                int action = listOfActions.at(j);
                policy[state][action] = 0.2; // uniform 5 action
            }
        }
    }
    data.updatePolicy(policy);
}

void RIRL::printNestedVector(vector<vector<int>> v){
    int counter = 0;
    ofstream myfile;
    myfile.open ("/home/shervin/Desktop/result.txt", ios::app);
    for(int i = 0 ; i < int(v.size()) ; i++){
        myfile << "<" << v.at(i).at(0) << " , " << v.at(i).at(1) << "> , ";
        cout << "<" << v.at(i).at(0) << " , " << v.at(i).at(1) << "> , ";
        counter ++;
    }
    cout << counter << endl;
    myfile << counter << endl;
}

void RIRL::printVector(vector<double> v){
    for(int i = 0 ; i < int(v.size()) ; i++){
        if(i == 0){
            cout << "< "<< v.at(i) << " , ";

        }else if(i > int(v.size())-1){
            cout << v.at(i) << " , ";

        }else{
            cout << v.at(i) << " >" << endl;
        }
    }
}

