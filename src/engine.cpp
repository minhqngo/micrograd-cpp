#include "engine.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace micrograd {

Value::Value(const Eigen::MatrixXd& data) : data(data) {
    grad = Eigen::MatrixXd::Zero(data.rows(), data.cols());
    _backward = []() {};
}

Value::Value(const double scalar) : data(Eigen::MatrixXd::Constant(1, 1, scalar)) {
    grad = Eigen::MatrixXd::Zero(1, 1);
    _backward = []() {};
}

void Value::set_self(const std::shared_ptr<Value>& ptr) {
    _self = ptr;
}

std::shared_ptr<Value> Value::get_self_ptr() const {
    auto ptr = _self.lock();
    if (ptr) {
        return ptr;
    }
    // If no self-pointer exists, create a new shared_ptr (for temporary values)
    return std::make_shared<Value>(*this);
}

Eigen::MatrixXd Value::broadcast_backward(const Eigen::MatrixXd& grad,
                                          int target_rows, int target_cols) {
    if (grad.rows() == target_rows && grad.cols() == target_cols) {
        return grad;
    }

    Eigen::MatrixXd result = grad;

    // Sum over broadcasting dimensions
    if (target_rows == 1 && grad.rows() > 1) {
        result = grad.colwise().sum();
    }
    if (target_cols == 1 && grad.cols() > 1) {
        result = result.rowwise().sum();
    }

    return result;
}

Value Value::operator+(const Value& other) const {
    Eigen::MatrixXd result;

    if (data.rows() == other.data.rows() && data.cols() == other.data.cols()) {
        result = data + other.data;
    } else if (other.data.rows() == 1 && data.cols() == other.data.cols()) {
        // Broadcast other (bias) across rows
        result = data.rowwise() + other.data.row(0);
    } else if (data.rows() == 1 && data.cols() == other.data.cols()) {
        // Broadcast self across rows
        result = other.data.rowwise() + data.row(0);
    } else {
        result = data + other.data;
    }

    Value out(result);
    out._op = "+";

    auto self_ptr = this->get_self_ptr();
    auto other_ptr = other.get_self_ptr();
    auto out_ptr = std::make_shared<Value>(out);
    out_ptr->set_self(out_ptr);

    out_ptr->_prev = {self_ptr, other_ptr};
    out_ptr->_backward = [self_ptr, other_ptr, out_ptr]() {
        self_ptr->grad += broadcast_backward(out_ptr->grad, self_ptr->data.rows(), self_ptr->data.cols());
        other_ptr->grad += broadcast_backward(out_ptr->grad, other_ptr->data.rows(), other_ptr->data.cols());
    };

    return *out_ptr;
}

Value Value::operator+(const double scalar) const {
    return *this + Value(scalar);
}

Value Value::operator*(const Value& other) const {
    Value out(data.array() * other.data.array());
    out._op = "*";

    auto self_ptr = this->get_self_ptr();
    auto other_ptr = other.get_self_ptr();
    auto out_ptr = std::make_shared<Value>(out);
    out_ptr->set_self(out_ptr);

    out_ptr->_prev = {self_ptr, other_ptr};
    out_ptr->_backward = [self_ptr, other_ptr, out_ptr]() {
        Eigen::MatrixXd grad_self = other_ptr->data.array() * out_ptr->grad.array();
        Eigen::MatrixXd grad_other = self_ptr->data.array() * out_ptr->grad.array();
        self_ptr->grad += broadcast_backward(grad_self, self_ptr->data.rows(), self_ptr->data.cols());
        other_ptr->grad += broadcast_backward(grad_other, other_ptr->data.rows(), other_ptr->data.cols());
    };

    return *out_ptr;
}

Value Value::operator*(const double scalar) const {
    return *this * Value(scalar);
}

Value Value::operator-(const Value& other) const {
    return *this + (other * -1.0);
}

Value Value::operator-(const double scalar) const {
    return *this + (-scalar);
}

Value Value::operator/(const Value& other) const {
    return *this * other.pow(-1.0);
}

Value Value::operator/(const double scalar) const {
    return *this * (1.0 / scalar);
}

Value Value::matmul(const Value& other) const {
    Value out(data * other.data);
    out._op = "@";

    auto self_ptr = this->get_self_ptr();
    auto other_ptr = other.get_self_ptr();
    auto out_ptr = std::make_shared<Value>(out);
    out_ptr->set_self(out_ptr);

    out_ptr->_prev = {self_ptr, other_ptr};
    out_ptr->_backward = [self_ptr, other_ptr, out_ptr]() {
        self_ptr->grad += out_ptr->grad * other_ptr->data.transpose();
        other_ptr->grad += self_ptr->data.transpose() * out_ptr->grad;
    };

    return *out_ptr;
}

Value Value::pow(double exponent) const {
    Value out(data.array().pow(exponent));
    out._op = "pow";

    auto self_ptr = this->get_self_ptr();
    auto out_ptr = std::make_shared<Value>(out);
    out_ptr->set_self(out_ptr);

    out_ptr->_prev = {self_ptr};
    out_ptr->_backward = [self_ptr, out_ptr, exponent]() {
        self_ptr->grad += ((exponent * self_ptr->data.array().pow(exponent - 1)) * out_ptr->grad.array()).matrix();
    };

    return *out_ptr;
}

Value Value::relu() const {
    Value out(data.array().max(0.0));
    out._op = "relu";

    auto self_ptr = this->get_self_ptr();
    auto out_ptr = std::make_shared<Value>(out);
    out_ptr->set_self(out_ptr);

    out_ptr->_prev = {self_ptr};
    out_ptr->_backward = [self_ptr, out_ptr]() {
        self_ptr->grad += ((out_ptr->data.array() > 0.0).cast<double>() * out_ptr->grad.array()).matrix();
    };

    return *out_ptr;
}

Value Value::sigmoid() const {
    Eigen::MatrixXd s = 1.0 / (1.0 + (-data.array()).exp());
    Value out(s);
    out._op = "sigmoid";

    auto self_ptr = this->get_self_ptr();
    auto out_ptr = std::make_shared<Value>(out);
    out_ptr->set_self(out_ptr);

    out_ptr->_prev = {self_ptr};
    out_ptr->_backward = [self_ptr, out_ptr, s]() {
        self_ptr->grad += ((s.array() * (1.0 - s.array())) * out_ptr->grad.array()).matrix();
    };

    return *out_ptr;
}

Value Value::transpose() const {
    Value out(data.transpose());
    out._op = "T";

    auto self_ptr = this->get_self_ptr();
    auto out_ptr = std::make_shared<Value>(out);
    out_ptr->set_self(out_ptr);

    out_ptr->_prev = {self_ptr};
    out_ptr->_backward = [self_ptr, out_ptr]() {
        self_ptr->grad += out_ptr->grad.transpose();
    };

    return *out_ptr;
}

Value Value::flatten() const {
    int orig_rows = data.rows();
    int orig_cols = data.cols();

    if (orig_cols <= 1) {
        return *this;
    }

    int flattened_cols = orig_rows * orig_cols;
    Eigen::MatrixXd flattened = Eigen::Map<const Eigen::MatrixXd>(data.data(), 1, flattened_cols);

    Value out(flattened);
    out._op = "flatten";

    auto self_ptr = this->get_self_ptr();
    auto out_ptr = std::make_shared<Value>(out);
    out_ptr->set_self(out_ptr);

    out_ptr->_prev = {self_ptr};
    out_ptr->_backward = [self_ptr, out_ptr, orig_rows, orig_cols]() {
        Eigen::MatrixXd reshaped = Eigen::Map<const Eigen::MatrixXd>(out_ptr->grad.data(), orig_rows, orig_cols);
        self_ptr->grad += reshaped;
    };

    return *out_ptr;
}

void Value::build_topo(std::shared_ptr<Value> v, std::set<std::shared_ptr<Value>>& visited,
                       std::vector<std::shared_ptr<Value>>& topo) {
    if (visited.find(v) == visited.end()) {
        visited.insert(v);
        for (auto& child : v->_prev) {
            build_topo(child, visited, topo);
        }
        topo.push_back(v);
    }
}

void Value::backward() {
    std::vector<std::shared_ptr<Value>> topo;
    std::set<std::shared_ptr<Value>> visited;

    auto self_ptr = this->get_self_ptr();
    build_topo(self_ptr, visited, topo);

    self_ptr->grad = Eigen::MatrixXd::Ones(data.rows(), data.cols());

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }

    // If this was a stack variable, copy gradients back
    if (!_self.lock()) {
        *this = *self_ptr;
    }
}

void Value::zero_grad() {
    grad = Eigen::MatrixXd::Zero(data.rows(), data.cols());
}

// ValuePtr implementations
ValuePtr::ValuePtr(const Eigen::MatrixXd& data) : ptr(std::make_shared<Value>(data)) {}

ValuePtr::ValuePtr(double scalar) : ptr(std::make_shared<Value>(scalar)) {}

ValuePtr ValuePtr::operator+(const ValuePtr& other) const {
    return {std::make_shared<Value>(*ptr + *other.ptr)};
}

ValuePtr ValuePtr::operator+(double scalar) const {
    return {std::make_shared<Value>(*ptr + scalar)};
}

ValuePtr ValuePtr::operator*(const ValuePtr& other) const {
    return {std::make_shared<Value>(*ptr * *other.ptr)};
}

ValuePtr ValuePtr::operator*(double scalar) const {
    return {std::make_shared<Value>(*ptr * scalar)};
}

ValuePtr ValuePtr::operator-(const ValuePtr& other) const {
    return {std::make_shared<Value>(*ptr - *other.ptr)};
}

ValuePtr ValuePtr::operator-(double scalar) const {
    return {std::make_shared<Value>(*ptr - scalar)};
}

ValuePtr ValuePtr::operator/(const ValuePtr& other) const {
    return {std::make_shared<Value>(*ptr / *other.ptr)};
}

ValuePtr ValuePtr::operator/(double scalar) const {
    return {std::make_shared<Value>(*ptr / scalar)};
}

ValuePtr ValuePtr::matmul(const ValuePtr& other) const {
    return {std::make_shared<Value>(ptr->matmul(*other.ptr))};
}

ValuePtr ValuePtr::pow(double exponent) const {
    return {std::make_shared<Value>(ptr->pow(exponent))};
}

ValuePtr ValuePtr::relu() const {
    return {std::make_shared<Value>(ptr->relu())};
}

ValuePtr ValuePtr::sigmoid() const {
    return {std::make_shared<Value>(ptr->sigmoid())};
}

ValuePtr ValuePtr::transpose() const {
    return {std::make_shared<Value>(ptr->transpose())};
}

ValuePtr ValuePtr::flatten() const {
    return {std::make_shared<Value>(ptr->flatten())};
}

} // namespace micrograd
