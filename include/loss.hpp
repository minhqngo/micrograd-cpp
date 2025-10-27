#pragma once

#include "engine.hpp"
#include "nn.hpp"
#include <Eigen/Dense>

namespace micrograd {

Eigen::MatrixXd softmax(const Value& x);

class CrossEntropyLoss : public Module {
public:
    CrossEntropyLoss() = default;
    Value forward(const Value& y_pred, const Eigen::VectorXi& y_true);
    std::vector<Value*> parameters() override { return {}; }
};

class MSELoss : public Module {
public:
    MSELoss() = default;
    Value forward(const Value& y_pred, const Value& y_true);
    std::vector<Value*> parameters() override { return {}; }
};

} // namespace micrograd
