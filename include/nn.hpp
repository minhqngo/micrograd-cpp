#pragma once

#include "engine.hpp"
#include <vector>
#include <string>

namespace micrograd {

class Module {
public:
    virtual ~Module() = default;
    virtual std::vector<Value*> parameters() = 0;
    virtual void zero_grad();
    void save_weights(const std::string& path);
    void load_weights(const std::string& path);
};

class Layer : public Module {
public:
    std::shared_ptr<Value> w;
    std::shared_ptr<Value> b;
    bool nonlin;

    Layer(int nin, int nout, bool nonlin = true);
    Value forward(const Value& x) const;
    std::vector<Value*> parameters() override;
};

class MLP : public Module {
public:
    std::vector<Layer> layers;

    MLP(int nin, const std::vector<int>& nouts);
    Value forward(const Value& x) const;
    std::vector<Value*> parameters() override;
};

} // namespace micrograd
