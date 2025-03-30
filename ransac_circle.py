import numpy as np
from sklearn.metrics import mean_squared_error

def fit_circle_least_squares(points):
    """
    Fits a circle to 2D points using least squares (Kåsa method).
    
    Args:
        points: (N, 2) array of 2D points.
    
    Returns:
        center: (2,) array (x, y).
        radius: float.
    """
    x, y = points[:, 0], points[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    c = np.linalg.lstsq(A, b, rcond=None)[0]  # Solve Ax = b
    center = c[:2]
    radius = np.sqrt(c[2] + np.sum(center**2))
    return center, radius

def ransac_circle(points, n_iter=100, threshold=0.1, min_inliers=10):
    """
    RANSAC for circle fitting in 2D.
    
    Args:
        points: (N, 2) array of 2D points.
        n_iter: Number of RANSAC iterations.
        threshold: Max distance to consider a point an inlier.
        min_inliers: Minimum inliers to accept a circle.
    
    Returns:
        best_center: (2,) array (x, y) or None if no fit found.
        best_radius: float or None.
        inliers: (M,) boolean array of inlier indices.
    """
    best_inliers = None
    best_center, best_radius = None, None
    
    for _ in range(n_iter):
        # Randomly sample 3 points
        sample = points[np.random.choice(len(points), 3, replace=False)]
        
        # Fit circle to the sample
        try:
            center, radius = fit_circle_least_squares(sample)
        except np.linalg.LinAlgError:
            continue  # Skip degenerate cases
        
        # Compute distances to hypothesized circle
        distances = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        
        # Count inliers
        inliers = distances < threshold
        n_inliers = np.sum(inliers)
        
        # Update best model
        if n_inliers >= min_inliers and (best_inliers is None or n_inliers > np.sum(best_inliers)):
            best_inliers = inliers
            best_center, best_radius = center, radius
    
    # Refit using all inliers (if any)
    if best_inliers is not None and np.sum(best_inliers) >= 3:
        best_center, best_radius = fit_circle_least_squares(points[best_inliers])
    
    return best_center, best_radius, best_inliers