# NeuralNet

A simple neural network implemented entirely from scratch in Rust.

This was just a fun weekend project for me to establish my understanding of neural networks by training to the canonical
MNIST dataset. All math is represented in vectorized form, which has the benefit of being much more difficult to
understand and much harder to verify... If anyone who deeply understands the math of backpropogation could give it a
once over to make sure I haven't swapped the order of my multiplications or transposed the wrong matrices that would be
extremely helpful (Just open an issue if you find anything).

Despite some uncertainty around the validity of my math, I'm able to train a neural network which identifies written
characters with approximately 88% accuracy. From what I've seen online I should probably getting better results, so I
will probably dedicate a few hours in the future to validating my math and increasing accuracy of the model.

