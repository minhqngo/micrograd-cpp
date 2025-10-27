#pragma once

#include "engine.hpp"
#include <vector>

namespace micrograd {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
};

class SGD : public Optimizer {
public:
    std::vector<Value*> parameters;
    double lr;

    SGD(const std::vector<Value*>& params, double learning_rate = 0.01);
    void step() override;
    void zero_grad() override;
};

class NesterovSGD : public Optimizer {
public:
    std::vector<Value*> parameters;
    double lr;
    double mu;
    std::vector<Eigen::MatrixXd> v;

    NesterovSGD(const std::vector<Value*>& params, double learning_rate = 0.01, double momentum = 0.9);
    void step() override;
    void zero_grad() override;
};

} // namespace micrograd
