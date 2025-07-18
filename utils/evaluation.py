from config import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def id_to_coor(pred: np.ndarray,
               width: int = config.env_config.WIDTH,
               length: int = config.env_config.LENGTH) -> pd.DataFrame:
    """
    Convert predicted cell IDs to corresponding coordinates.

    Args:
        pred (numpy.ndarray): Array of predicted cell IDs.
        width (int): Width of the environment grid. Defaults to config.env_config.WIDTH.
        length (int): Length of the environment grid. Defaults to config.env_config.LENGTH.

    Returns:
        pandas.DataFrame: A DataFrame containing the corresponding 'x' and 'y' coordinates,
        as well as the original 'cell' ID for each prediction.
    """
    x = pred//length
    y = pred-x*length + 0.5
    x = x+0.5
    df = pd.DataFrame({'x': x, 'y': y, 'cell': pred})
    return df


def coor_to_id(pred: np.ndarray,
               width: int = config.env_config.WIDTH,
               length: int = config.env_config.LENGTH) -> pd.DataFrame:
    """
    Convert coordinates to corresponding cell IDs.

    Args:
        pred (numpy.ndarray): Array of coordinates with shape (N, 2),
            where each row represents an (x, y) pair.
        width (int): Width of the environment grid. Defaults to config.env_config.WIDTH.
        length (int): Length of the environment grid. Defaults to config.env_config.LENGTH.

    Returns:
        pandas.DataFrame: A DataFrame containing the original 'x' and 'y' coordinates,
        as well as the computed 'cell' ID for each point.
    """
    ids = (pred[:, 0]//1)*length + (pred[:, 1]//1) + 1
    df = pd.DataFrame(
        {'x': pred[:, 0], 'y': pred[:, 1], 'cell': np.array(ids, dtype=np.int16)})
    return df


def compute_distances_error(df_true: pd.DataFrame,
                            df_pred: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Compute distance errors between true and predicted coordinates.

    Args:
        df_true (pandas.DataFrame): DataFrame containing ground truth coordinates.
            Must include columns 'x' and 'y'.
        df_pred (pandas.DataFrame): DataFrame containing predicted coordinates.
            Must include columns 'x' and 'y'.

    Returns:
        Tuple[float, float, float]: A tuple containing:
            - The mean Euclidean distance error.
            - The maximum Euclidean distance error.
            - The minimum Euclidean distance error.
    """
    N = len(df_pred)
    distances = ((df_true['x']-df_pred['x'])**2 +
                 (df_true['y']-df_pred['y'])**2)**0.5
    q25, q50, q75 = np.percentile(distances, [25, 50, 75])
    # print(f'distance:{sorted(distances.values)}')
    print(
        f'index of largest error:{list(distances).index(sorted(distances.values)[-1])}')
    return distances.mean(), max(distances), min(distances), q25, q50, q75  # , distances.var()


def draw_distribution_of_dist_err(df_true: pd.DataFrame,
                                  df_pred: pd.DataFrame,
                                  model_name: str,
                                  dir_path: str = "./images/") -> None:
    """
    Plot and save a histogram of distance errors between predicted and true coordinates.

    Args:
        df_true (pandas.DataFrame): DataFrame containing ground truth coordinates.
            Must include columns 'x' and 'y'.
        df_pred (pandas.DataFrame): DataFrame containing predicted coordinates.
            Must include columns 'x' and 'y'.
        model_name (str): The name of the model to be used in the plot title and filename.
        dir_path (str): Directory path to save the plot image. Defaults to "../images/".

    Returns:
        None
    """
    distances = ((df_true['x']-df_pred['x'])**2 +
                 (df_true['y']-df_pred['y'])**2)**0.5
    plt.plot()
    plt.hist(distances, bins=100, range=(0, 25), color='gray', alpha=0.7)
    plt.xlabel("error")
    plt.ylabel("freq")
    plt.xlim(0, 25)
    plt.ylim(0, 70)
    plt.title(f'{model_name} distance error histogram')
    plt.savefig(dir_path+f'{model_name}_distance_error.png')
    plt.close()
    return
