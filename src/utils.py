import numpy as np
from scipy.spatial.transform import Rotation as R


def identity_transform() -> np.ndarray:
    """Returns an identity transform

    Returns:
        np.ndarray: 4x4 homogeneous transform matrix
    """
    return np.eye(4, 4, dtype=np.float64)


def create_transform_from_translation_and_yaw(translation: tuple[float, float, float],
                                              yaw_degrees: float = 0.0) -> np.ndarray:
    """Create a transformation matrix from a translation vector
    and a CCW rotation around the z-Axis.

    Args:
        translation (tuple[float, float]): translation vector in [m]
        yaw_degrees (float, optional): CCW rotation in degees around the z-Axis.
            Defaults to 0.0.

    Returns:
        np.ndarray: 4x4 homogeneous transform matrix
    """
    T = identity_transform()

    r = R.from_euler('z', yaw_degrees, degrees=True)
    T[:3, :3] = r.as_matrix()

    T[:3, 3] = np.array(translation)
    return T
