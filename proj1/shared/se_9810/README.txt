SE-ResNet-164 / RICAP test probabilities

se_probs.npy: float32, shape (2000, 10). Load with np.load("se_probs.npy", allow_pickle=False).
Rows: test_0000.png through test_1999.png, in that order.
Columns 0-9: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
Inference: average softmax probabilities from original and horizontally flipped images, FP32.
Model: 2,488,378 parameters; trained from scratch on all 50,000 permitted CIFAR-10 training images for 1,800 epochs.
The corresponding 45k/5k validation recipe scored 98.10%; that is a local validation score.
This is the original SE/RICAP model, before the SAM continuation.
No external training data or pretrained weights.
