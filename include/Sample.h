#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <glog/logging.h>
// include headers that implement a archive in simple text format
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
using namespace std;

#ifndef SAMPLE_H
#define SAMPLE_H

class Sample{

public:
    Sample();
    ~Sample();
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
      ar & values;
      ar & p;
    }

    vector<double> values;
    double p; //make uniform

    void init_rand(vector<double> *low_limit, vector<double> *high_limit);
    Sample combine(vector<double> second);
    size_t size();
    string str();
};

#endif
