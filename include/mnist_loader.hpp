#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace micrograd {

class MNISTLoader {
public:
    Eigen::MatrixXd images;
    Eigen::VectorXi labels;
    int num_images;
    int image_rows;
    int image_cols;

    MNISTLoader() = default;
    bool load(const std::string& images_path, const std::string& labels_path);
    void get_batch(int batch_idx, int batch_size, Eigen::MatrixXd& batch_images, Eigen::VectorXi& batch_labels);
    int get_num_batches(int batch_size) const;

private:
    static int reverse_int(int i);
};

} // namespace micrograd
