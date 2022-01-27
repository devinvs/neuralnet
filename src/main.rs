use std::fs::File;
use std::io::Read;

use neuralnet::matrix::Matrix;

fn main() {
    // Load inputs
    let test_images = load_images("./data/t10k-images-idx3-ubyte");
    let test_labels = load_labels("./data/t10k-labels-idx1-ubyte");

    let train_images = load_images("./data/train-images-idx3-ubyte");
    let train_labels = load_labels("./data/train-labels-idx1-ubyte");

    println!("Loaded Dataset: {} {} {} {}", test_images.len(), test_labels.len(), train_images.len(), train_labels.len());


    // Create Neural Network with random initialization
    let mut w1 = Matrix::<f32, 128, 784>::random();
    let mut w2 = Matrix::<f32, 64, 128>::random();
    let mut w3 = Matrix::<f32, 10, 64>::random();

    let mut b1 = Matrix::<f32, 128, 1>::new();
    let mut b2 = Matrix::<f32, 64, 1>::new();
    let mut b3 = Matrix::<f32, 10, 1>::new();

    let r = 0.00100;

    // Train the neural network
    for i in 0..20 {
        let mut error = 0.0;
        for (&x, y) in train_images.iter().zip(train_labels.iter()) {
            let z1 = w1*x + b1;
            let a1 = z1.apply(sigmoid);
            let z2 = w2*a1 + b2;
            let a2 = z2.apply(sigmoid);
            let z3 = w3*a2 + b3;
            let a3 = z3.apply(sigmoid);

            let pred =a3.max_index();
            let actual = y.max_index();

            if pred != actual {
                error += 1.0;
            }

            let diff = a3-*y;

            // Working!!!!!!!!!
            let da3 = diff * 2.0;
            let dz3 = a3.apply(sigmoid_d).hadamard(da3);
            let dw3 = dz3 * a2.transpose();
            let db3 = dz3;

            let da2 = w3.transpose() * dz3;
            let dz2 = a2.apply(sigmoid_d).hadamard(da2);
            let dw2 = dz2 * a1.transpose();
            let db2 = dz2;

            let da1 = w2.transpose() * dz2;
            let dz1 = a1.apply(sigmoid_d).hadamard(da1);
            let dw1 = dz1 * x.transpose();
            let db1 = dz1;

            w3 -= dw3 * r;
            w2 -= dw2 * r;
            w1 -= dw1 * r;

            b3 -= db3 * r;
            b2 -= db2 * r;
            b1 -= db1 * r;
        }

        println!("epoch: {i}\t{error}");
    }

    // Test against test datasets, this is where it matters
    let mut error = 1.0;
    for (&x, y) in test_images.iter().zip(test_labels.iter()) {
        let z1 = w1*x + b1;
        let a1 = z1.apply(sigmoid);
        let z2 = w2*a1 + b2;
        let a2 = z2.apply(sigmoid);
        let z3 = w3*a2 + b3;
        let a3 = z3.apply(sigmoid);

        let pred =a3.max_index();
        let actual = y.max_index();

        if pred != actual {
            error += 1.0;
        }
    }

    println!("Final accuracy: {} correct, {} incorrect, {}%", 10000.0-error, error, (10000.0-error)/100.0);
}


fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_d(x: f32) -> f32 {
    sigmoid(x) * (1.0-sigmoid(x))
}

fn load_labels(path: &str) -> Vec<Matrix<f32, 10, 1>> {
    let mut file = File::open(path).unwrap();
    let mut int_buffer = [0; 4];

    // Check magic bytes
    file.read_exact(&mut int_buffer).unwrap();
    let magic_num = u32::from_be_bytes(int_buffer);
    assert_eq!(2049, magic_num);

    // Get length of data
    file.read_exact(&mut int_buffer).unwrap();
    let size = u32::from_be_bytes(int_buffer);

    // List of properly formatted labels
    let mut labels = Vec::<Matrix<f32, 10, 1>>::with_capacity(size as usize);
    let mut buffer = [0; 4096];

    while let Ok(bytes_read) = file.read(&mut buffer) {
        if bytes_read==0 {break;}

        for i in 0..bytes_read {
            let label = u8::from_be(buffer[i]);

            assert!(label <= 9);
            let mut m = Matrix::<f32, 10, 1>::fill_with(0.0);
            m.set(label as usize, 0, 1.0);
            labels.push(m);
        }
    }

    labels
}

fn load_images(path: &str) -> Vec<Matrix<f32, 784, 1>> {
    let mut file = File::open(path).unwrap();
    let mut int_buffer = [0; 4];

    // Check magic bytes
    file.read_exact(&mut int_buffer).unwrap();
    let magic_num = u32::from_be_bytes(int_buffer);
    assert_eq!(2051, magic_num);

    // Get length of data
    file.read_exact(&mut int_buffer).unwrap();
    let size = u32::from_be_bytes(int_buffer);


    // List of images with bias parameter
    let mut images = Vec::<Matrix<f32, 784, 1>>::with_capacity(size as usize);
    let mut buffer = [0; 4704]; // 6 images

    while let Ok(bytes_read) = file.read(&mut buffer) {
        if bytes_read == 0 {break;}

        for i in 0..bytes_read/784 {
            let mut m = Matrix::<f32, 784, 1>::fill_with(1.0);

            // Copy into matrix
            for j in 0..784 {
                m.set(j, 0, buffer[i*784+j] as f32 / 255.0)
            }

            images.push(m);
        }
    }

    images
}
