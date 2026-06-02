from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarchenkoPasturLaw:
    """Marchenko-Pastur reference law for sample covariance/correlation spectra.

    Parameters use the common random-matrix notation:
    - `num_assets`: cross-sectional dimension `N`
    - `sample_size`: number of observations `T`
    - `variance`: population variance scale, equal to 1.0 for standardized returns

    The bulk support is defined by:
    `(variance * (1 - sqrt(q))**2, variance * (1 + sqrt(q))**2)`
    where `q = N / T`.
    """

    num_assets: int
    sample_size: int
    variance: float = 1.0

    @property
    def aspect_ratio(self) -> float:
        return float(self.num_assets) / float(self.sample_size)

    @property
    def lambda_minus(self) -> float:
        q = self.aspect_ratio
        return float(self.variance * (1.0 - np.sqrt(q)) ** 2)

    @property
    def lambda_plus(self) -> float:
        q = self.aspect_ratio
        return float(self.variance * (1.0 + np.sqrt(q)) ** 2)

    @property
    def has_point_mass_at_zero(self) -> bool:
        return self.aspect_ratio > 1.0

    @property
    def zero_mass_weight(self) -> float:
        if not self.has_point_mass_at_zero:
            return 0.0
        return 1.0 - (1.0 / self.aspect_ratio)

    def pdf(self, eigenvalues: np.ndarray | list[float]) -> np.ndarray:
        """Evaluate the continuous MP density on the provided eigenvalue grid."""
        grid = np.asarray(eigenvalues, dtype=float)
        density = np.zeros_like(grid, dtype=float)
        q = self.aspect_ratio
        if q <= 0.0:
            return density

        support_mask = (grid >= self.lambda_minus) & (grid <= self.lambda_plus)
        if not np.any(support_mask):
            return density

        x = grid[support_mask]
        numerator = np.sqrt(np.clip((self.lambda_plus - x) * (x - self.lambda_minus), 0.0, None))
        denominator = 2.0 * np.pi * self.variance * q * x
        density[support_mask] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(x, dtype=float),
            where=denominator > 0.0,
        )
        return density

    def density_grid(self, *, num_points: int = 512, padding: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """Build a plotting grid slightly wider than the theoretical support."""
        width = max(self.lambda_plus - self.lambda_minus, 1e-12)
        lo = max(0.0, self.lambda_minus - padding * width)
        hi = self.lambda_plus + padding * width
        grid = np.linspace(lo, hi, num_points)
        return grid, self.pdf(grid)

    def cdf_grid(self, *, num_points: int = 2048, padding: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Numerically integrate the continuous MP density to obtain a CDF grid."""
        grid, density = self.density_grid(num_points=num_points, padding=padding)
        cdf = np.zeros_like(grid, dtype=float)
        if len(grid) > 1:
            dx = np.diff(grid)
            trapezoids = 0.5 * (density[:-1] + density[1:]) * dx
            cdf[1:] = np.cumsum(trapezoids)
        total_mass = cdf[-1] if len(cdf) else 0.0
        if total_mass > 0.0:
            cdf = cdf / total_mass
        return grid, np.clip(cdf, 0.0, 1.0)

    def quantile_curve(self, ranks: np.ndarray | list[int], *, num_points: int = 2048) -> np.ndarray:
        """Map descending eigenvalue ranks to the theoretical MP bulk quantiles."""
        rank_array = np.asarray(ranks, dtype=float)
        if rank_array.ndim != 1:
            raise ValueError('ranks must be a one-dimensional array.')
        if np.any(rank_array <= 0):
            raise ValueError(f'ranks must be strictly positive (got {rank_array}).')

        total_count = max(self.num_assets, 1)
        # Descending scree ranks correspond to upper-tail quantiles.
        probs = 1.0 - ((rank_array - 0.5) / total_count)
        probs = np.clip(probs, 0.0, 1.0)

        if self.has_point_mass_at_zero:
            continuous_mass = 1.0 - self.zero_mass_weight
            probs = np.where(probs <= self.zero_mass_weight, 0.0, (probs - self.zero_mass_weight) / max(continuous_mass, 1e-12))

        grid, cdf = self.cdf_grid(num_points=num_points)
        return np.interp(probs, cdf, grid, left=grid[0], right=grid[-1])



def marchenko_pastur_law(num_assets: int, sample_size: int, *, variance: float = 1.0) -> MarchenkoPasturLaw:
    if num_assets <= 0:
        raise ValueError(f'num_assets must be strictly positive (got {num_assets}).')
    if sample_size <= 0:
        raise ValueError(f'sample_size must be strictly positive (got {sample_size}).')
    if variance <= 0.0:
        raise ValueError(f'variance must be strictly positive (got {variance}).')
    return MarchenkoPasturLaw(num_assets=num_assets, sample_size=sample_size, variance=float(variance))
