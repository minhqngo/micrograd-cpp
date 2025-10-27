#include "mnist_loader.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace micrograd {

int MNISTLoader::reverse_int(const int i) {
    const unsigned char c1 = i & 255;
    const unsigned char c2 = (i >> 8) & 255;
    const unsigned char c3 = (i >> 16) & 255;
    const unsigned char c4 = (i >> 24) & 255;
    return (static_cast<int>(c1) << 24) + (static_cast<int>(c2) << 16) + (static_cast<int>(c3) << 8) + c4;
}

bool MNISTLoader::load(const std::string& images_path, const std::string& labels_path) {
    // Load images
    std::ifstream image_file(images_path, std::ios::binary);
    if (!image_file.is_open()) {
        std::cerr << "Cannot open image file: " << images_path << std::endl;
        return false;
    }

    int magic_number = 0;
    image_file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    magic_number = reverse_int(magic_number);

    if (magic_number != 2051) {
        std::cerr << "Invalid MNIST image file!" << std::endl;
        return false;
    }

    image_file.read(reinterpret_cast<char *>(&num_images), sizeof(num_images));
    num_images = reverse_int(num_images);
    image_file.read(reinterpret_cast<char *>(&image_rows), sizeof(image_rows));
    image_rows = reverse_int(image_rows);
    image_file.read(reinterpret_cast<char *>(&image_cols), sizeof(image_cols));
    image_cols = reverse_int(image_cols);

    int image_size = image_rows * image_cols;
    images = Eigen::MatrixXd(num_images, image_size);

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel = 0;
            image_file.read(reinterpret_cast<char *>(&pixel), sizeof(pixel));
            images(i, j) = static_cast<double>(pixel);
        }
    }

    image_file.close();

    // Load labels
    std::ifstream label_file(labels_path, std::ios::binary);
    if (!label_file.is_open()) {
        std::cerr << "Cannot open label file: " << labels_path << std::endl;
        return false;
    }

    label_file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    magic_number = reverse_int(magic_number);

    if (magic_number != 2049) {
        std::cerr << "Invalid MNIST label file!" << std::endl;
        return false;
    }

    int num_labels = 0;
    label_file.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));
    num_labels = reverse_int(num_labels);

    if (num_labels != num_images) {
        std::cerr << "Number of labels does not match number of images!" << std::endl;
        return false;
    }

    labels = Eigen::VectorXi(num_labels);

    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        label_file.read(reinterpret_cast<char *>(&label), sizeof(label));
        labels(i) = static_cast<int>(label);
    }

    label_file.close();

    std::cout << "Loaded " << num_images << " images (" << image_rows << "x" << image_cols << ")" << std::endl;
    return true;
}

void MNISTLoader::get_batch(int batch_idx, int batch_size, Eigen::MatrixXd& batch_images, Eigen::VectorXi& batch_labels) {
    int start_idx = batch_idx * batch_size;
    int end_idx = std::min(start_idx + batch_size, num_images);
    int actual_batch_size = end_idx - start_idx;

    batch_images = images.block(start_idx, 0, actual_batch_size, images.cols());
    batch_labels = labels.segment(start_idx, actual_batch_size);

    batch_images /= 255.0;
}

int MNISTLoader::get_num_batches(const int batch_size) const {
    return (num_images + batch_size - 1) / batch_size;
}

} // namespace micrograd
