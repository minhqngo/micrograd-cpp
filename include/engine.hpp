#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <functional>
#include <set>

namespace micrograd {

class Value {
public:
    Eigen::MatrixXd data;
    Eigen::MatrixXd grad;

    Value(const Eigen::MatrixXd& data);
    Value(double scalar);

    // Self-pointer for graph connectivity
    void set_self(const std::shared_ptr<Value>& ptr);
    std::shared_ptr<Value> get_self_ptr() const;

    // Operations
    Value operator+(const Value& other) const;
    Value operator+(double scalar) const;
    Value operator*(const Value& other) const;
    Value operator*(double scalar) const;
    Value operator-(const Value& other) const;
    Value operator-(double scalar) const;
    Value operator/(const Value& other) const;
    Value operator/(double scalar) const;
    Value matmul(const Value& other) const;
    Value pow(double exponent) const;
    Value relu() const;
    Value sigmoid() const;
    Value transpose() const;
    Value flatten() const;

    // Backward propagation
    void backward();
    void zero_grad();

    // Shape utilities
    int rows() const { return data.rows(); }
    int cols() const { return data.cols(); }

    std::vector<std::shared_ptr<Value>> _prev;
    std::function<void()> _backward;
    std::string _op;
    std::weak_ptr<Value> _self;

private:
    void build_topo(std::shared_ptr<Value> v, std::set<std::shared_ptr<Value>>& visited,
                    std::vector<std::shared_ptr<Value>>& topo);

    static Eigen::MatrixXd broadcast_backward(const Eigen::MatrixXd& grad,
                                              int target_rows, int target_cols);

    friend class ValuePtr;
};

// Smart pointer wrapper for Value
class ValuePtr {
public:
    std::shared_ptr<Value> ptr;

    ValuePtr(const Eigen::MatrixXd& data);
    ValuePtr(double scalar);
    ValuePtr(std::shared_ptr<Value> p) : ptr(p) {}

    Value& operator*() { return *ptr; }
    const Value& operator*() const { return *ptr; }
    Value* operator->() { return ptr.get(); }
    const Value* operator->() const { return ptr.get(); }

    ValuePtr operator+(const ValuePtr& other) const;
    ValuePtr operator+(double scalar) const;
    ValuePtr operator*(const ValuePtr& other) const;
    ValuePtr operator*(double scalar) const;
    ValuePtr operator-(const ValuePtr& other) const;
    ValuePtr operator-(double scalar) const;
    ValuePtr operator/(const ValuePtr& other) const;
    ValuePtr operator/(double scalar) const;
    ValuePtr matmul(const ValuePtr& other) const;
    ValuePtr pow(double exponent) const;
    ValuePtr relu() const;
    ValuePtr sigmoid() const;
    ValuePtr transpose() const;
    ValuePtr flatten() const;
};

} // namespace micrograd
