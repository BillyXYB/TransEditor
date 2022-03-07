import numpy as np


def linear_interpolate(latent_code,
                       boundary,
                       start_distance=-100,
                       end_distance=100,
                       steps=10):
    """Manipulates the given latent code with respect to a particular boundary.
  Basically, this function takes a latent code and a boundary as inputs, and
  outputs a collection of manipulated latent codes. For example, let `steps` to
  be 10, then the input `latent_code` is with shape [1, latent_space_dim], input
  `boundary` is with shape [1, latent_space_dim] and unit norm, the output is
  with shape [10, latent_space_dim]. The first output latent code is
  `start_distance` away from the given `boundary`, while the last output latent
  code is `end_distance` away from the given `boundary`. Remaining latent codes
  are linearly interpolated.
  Input `latent_code` can also be with shape [1, num_layers, latent_space_dim]
  to support W+ space in Style GAN. In this case, all features in W+ space will
  be manipulated same as each other. Accordingly, the output will be with shape
  [10, num_layers, latent_space_dim].
  NOTE: Distance is sign sensitive.
  Args:
    latent_code: The input latent code for manipulation.
    boundary: The semantic boundary as reference.
    start_distance: The distance to the boundary where the manipulation starts.
      (default: -3.0)
    end_distance: The distance to the boundary where the manipulation ends.
      (default: 3.0)
    steps: Number of steps to move the latent code from start position to end
      position. (default: 10)
  """
    assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
            len(boundary.shape) == 2 and
            boundary.shape[1] == latent_code.shape[-1])

    linspace = np.linspace(start_distance, end_distance, steps)
    if len(latent_code.shape) == 2:
        linspace = linspace - latent_code.dot(boundary.T)
        linspace = linspace.reshape(-1, 1).astype(np.float32)
        return latent_code + linspace * boundary
    if len(latent_code.shape) == 3:
        linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
        return latent_code + linspace * boundary.reshape(1, 1, -1)
    raise ValueError(f'Input `latent_code` should be with shape '
                     f'[1, latent_space_dim] or [1, N, latent_space_dim] for '
                     f'W+ space in Style GAN!\n'
                     f'But {latent_code.shape} is received.')
