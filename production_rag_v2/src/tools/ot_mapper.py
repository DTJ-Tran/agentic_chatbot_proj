import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.solvers import linear
import numpy as np
from typing import List

class OTTMapper:
    """
    Experimental Optimal Transport Mapper for Domain Adaptation.
    Learns a linear mapping from 384D (Local FastEmbed) to 4096D (API Embeddings).
    """
    def __init__(self):
        self.W = None  # The learned linear projection matrix (384 x 4096)
        
    def _pad_to_dim(self, X: jnp.ndarray, target_dim: int) -> jnp.ndarray:
        """Pads an array with zeros to match a target dimension."""
        curr_dim = X.shape[1]
        if curr_dim >= target_dim:
            return X
        padding = jnp.zeros((X.shape[0], target_dim - curr_dim))
        return jnp.concatenate([X, padding], axis=1)

    def train_alignment(self, X_src: np.ndarray, Y_tgt: np.ndarray, reg: float = 0.1):
        """
        Trains the alignment matrix W using Optimal Transport.
        """
        print("🧠 [OTTMapper] Training Optimal Transport Barycentric Map...")
        X = jnp.array(X_src)
        Y = jnp.array(Y_tgt)
        
        N = X.shape[0]
        
        # 1. Pad X temporarily to compute OT cost matrix
        X_padded = self._pad_to_dim(X, Y.shape[1])
        
        # 2. Setup OT geometry
        geom = pointcloud.PointCloud(X_padded, Y, epsilon=0.05)
        
        # 3. Solve OT using Sinkhorn
        a = jnp.ones(N) / N
        b = jnp.ones(N) / N
        out = jax.jit(linear.solve)(geom, a, b)
        
        P = out.matrix
        
        # 4. Compute Barycentric Projection
        row_sum = jnp.sum(P, axis=1, keepdims=True)
        row_sum = jnp.where(row_sum == 0, 1.0, row_sum)
        Y_bary = jnp.dot(P, Y) / row_sum
        
        # 5. Fit an out-of-sample Linear Map (Ridge Regression)
        I = jnp.eye(X.shape[1])
        X_T_X = jnp.dot(X.T, X) + reg * I
        X_T_Y = jnp.dot(X.T, Y_bary)
        
        self.W = jnp.linalg.solve(X_T_X, X_T_Y)
        print(f"✅ [OTTMapper] Alignment W trained. Shape: {self.W.shape}")

    def project(self, vector_src: List[float]) -> List[float]:
        """
        Projects a single 384D local embedding to the 4096D space.
        """
        if self.W is None:
            raise ValueError("OTTMapper is not trained. Please call train_alignment first.")
        
        x = jnp.array([vector_src])
        y_proj = jnp.dot(x, self.W)
        
        return np.array(y_proj[0]).tolist()
