#include "loss.hpp"
#include <cmath>
#include <iostream>

namespace micrograd {

const double EPS = 1e-12;

Eigen::MatrixXd softmax(const Value& x) {
    Eigen::MatrixXd result(x.data.rows(), x.data.cols());

    for (int i = 0; i < x.data.rows(); ++i) {
        double max_val = x.data.row(i).maxCoeff();
        Eigen::RowVectorXd exp_vals = (x.data.row(i).array() - max_val).exp();
        double sum = exp_vals.sum();
        result.row(i) = exp_vals / sum;
    }

    return result;
}

Value CrossEntropyLoss::forward(const Value& y_pred, const Eigen::VectorXi& y_true) {
    Eigen::MatrixXd probs = softmax(y_pred);
    probs = probs.array().max(EPS).min(1.0 - EPS);

    int n_samples = y_pred.data.rows();
    int n_classes = y_pred.data.cols();

    // Create one-hot encoding
    Eigen::MatrixXd true_labels_oh = Eigen::MatrixXd::Zero(n_samples, n_classes);
    for (int i = 0; i < n_samples; ++i) {
        true_labels_oh(i, y_true(i)) = 1.0;
    }

    // Compute loss
    double loss_val = -(true_labels_oh.array() * probs.array().log()).sum() / n_samples;

    Value out(Eigen::MatrixXd::Constant(1, 1, loss_val));
    out._op = "CELoss";

    auto y_pred_ptr = y_pred.get_self_ptr();
    auto out_ptr = std::make_shared<Value>(out);
    out_ptr->set_self(out_ptr);

    out_ptr->_prev = {y_pred_ptr};
    out_ptr->_backward = [y_pred_ptr, out_ptr, probs, true_labels_oh, n_samples]() {
        Eigen::MatrixXd grad = (probs - true_labels_oh) / n_samples;
        y_pred_ptr->grad += grad * out_ptr->grad(0, 0);
    };

    return *out_ptr;
}

Value MSELoss::forward(const Value& y_pred, const Value& y_true) {
    double loss_val = (y_pred.data - y_true.data).array().square().mean();

    Value out(Eigen::MatrixXd::Constant(1, 1, loss_val));
    out._op = "MSELoss";

    auto y_pred_ptr = y_pred.get_self_ptr();
    auto y_true_ptr = y_true.get_self_ptr();
    auto out_ptr = std::make_shared<Value>(out);
    out_ptr->set_self(out_ptr);

    out_ptr->_prev = {y_pred_ptr};
    out_ptr->_backward = [y_pred_ptr, y_true_ptr, out_ptr]() {
        int size = y_pred_ptr->data.size();
        Eigen::MatrixXd grad = 2.0 * (y_pred_ptr->data - y_true_ptr->data) / size;
        y_pred_ptr->grad += grad * out_ptr->grad(0, 0);
    };

    return *out_ptr;
}

} // namespace micrograd
