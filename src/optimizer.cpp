#include "optimizer.hpp"

namespace micrograd {

SGD::SGD(const std::vector<Value*>& params, double learning_rate)
    : parameters(params), lr(learning_rate) {}

void SGD::step() {
    for (auto* p : parameters) {
        p->data -= lr * p->grad;
    }
}

void SGD::zero_grad() {
    for (auto* p : parameters) {
        p->zero_grad();
    }
}

NesterovSGD::NesterovSGD(const std::vector<Value*>& params, double learning_rate, double momentum)
    : parameters(params), lr(learning_rate), mu(momentum) {
    v.resize(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        v[i] = Eigen::MatrixXd::Zero(params[i]->data.rows(), params[i]->data.cols());
    }
}

void NesterovSGD::step() {
    for (size_t i = 0; i < parameters.size(); ++i) {
        Eigen::MatrixXd v_prev = v[i];
        v[i] = mu * v[i] - lr * parameters[i]->grad;
        parameters[i]->data += -mu * v_prev + (1.0 + mu) * v[i];
    }
}

void NesterovSGD::zero_grad() {
    for (auto* p : parameters) {
        p->zero_grad();
    }
}

} // namespace micrograd
