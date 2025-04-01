# Gravitational Lens Classification

## Strategies Used in Both Approaches

### 1. Weighted Random Sampling
To mitigate the effects of class imbalance, both approaches employ `WeightedRandomSampler`. This ensures that during training, each class is sampled with an appropriate probability, preventing the model from being biased toward the majority class.

## Distinct Strategies in Each Attempt

### Approach 1
- **Uses Weighted Random Sampling** to address class imbalance.
- **Employs Focal Loss with γ = 2** to reduce the impact of easy-to-classify examples and focus on harder samples.
- **Does not use class weights in the loss function**, instead relying on sampling strategies.
- **Applies Data Augmentation** to introduce variability in the minority class, improving generalization.

### Approach 2
- **Uses Weighted Random Sampling** to balance class representation.
- **Incorporates Class Weights in the Loss Function** to penalize misclassification of the minority class more strongly.
- **Applies Data Augmentation**, similar to Approach 1, to improve generalization.


