""" PyTorch Network Training Demo
Trains a simple CNN to make velocity fields incompressible.
This script runs for a certain number of steps before saving the trained network and halting.
"""
from phi.tf.flow import *


# TORCH.set_default_device('GPU')
net = u_net(2, 2)
optimizer = keras.optimizers.Adam(1e-3)

prediction = CenteredGrid((0, 0), extrapolation.BOUNDARY, x=64, y=64)
prediction_div = CenteredGrid(0, 0, x=64, y=64)
viewer = view(play=False, namespace=globals(), select='batch')

for step in viewer.range(100):
    # Load or generate training data
    data = CenteredGrid(Noise(batch(batch=8), channel(vector=2)), extrapolation.BOUNDARY, x=64, y=64)
    # Initialize optimizer
    with tf.GradientTape() as tape:
        # Prediction
        prediction = field.native_call(net, data)  # calls net with shape (BATCH_SIZE, channels, spatial...)
        # Simulation
        prediction_div = field.divergence(prediction)
        # Define loss and compute gradients
        loss = field.l2_loss(prediction_div) + field.l2_loss(prediction - data)
        gradients = tape.gradient(loss.mean, net.trainable_variables)
    viewer.log_scalars(loss=loss.mean, div=field.mean(abs(prediction_div)).mean, distance=math.vec_abs(field.mean(abs(prediction - data))).mean)
    # Update weights
    optimizer.apply_gradients(zip(gradients, net.trainable_variables))


net.save_weights('tf_net.h5')
viewer.info("Network saved.")

# To load the network: net.load_state_dict(torch.load('torch_net.pth'))
