from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray


class ProbabilityModel(ABC):
    """
    Abstract base class for probability models used in analog sampling.

    This class defines the interface for probability models that determine
    the likelihood of selecting an analog based on its similarity with the true next
    state. In some derived classes, restrictions on the coordinates of the candidate
    samples are imposed and probabilities are zero-ed if they aren't fulfilled.
    """

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the probability model.

        Parameters
        ----------
        *args : Any
            Positional arguments for model initialization.
        **kwargs : Any
            Keyword arguments for model initialization.
        """
        pass

    @abstractmethod
    def unnormalized_log_probability(
        self,
        coords_s_next: xr.Dataset,
        coords_candidates: xr.Dataset,
        similarities: NDArray,
    ) -> NDArray:
        """
        Compute unnormalized log probabilities for candidate analogs.

        Parameters
        ----------
        coords_s_next : xr.Dataset
            Coordinates of the true next state.
        coords_candidates : xr.Dataset
            Coordinates of candidate states.
        similarities : NDArray
            Similarity values for candidate states.

        Returns
        -------
        NDArray
            Unnormalized log probabilities for each candidate.
        """
        pass

    def sample(
        self,
        rng: np.random.Generator,
        size: int | tuple[int, ...],
        similarities: NDArray,
        coords_s_next: xr.Dataset,
        coords_candidates: xr.Dataset,
    ) -> NDArray:
        """
        Sample analog states using the Gumbel-max trick.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.
        size : int or tuple of int
            Number of samples to generate.
        similarities : NDArray
            Similarity values for candidate states.
        coords_s_next : xr.Dataset
            Coordinates of the true next state.
        coords_candidates : xr.Dataset
            Coordinates of candidate states.

        Returns
        -------
        NDArray
            Indices of sampled analog states.
        """
        unnormalized_logp = self.unnormalized_log_probability(
            similarities=similarities,
            coords_s_next=coords_s_next,
            coords_candidates=coords_candidates,
        )

        return gumbel_max_sample(
            unnormalized_logp=unnormalized_logp, rng=rng, size=size
        )


class NormalProbabilityModel(ProbabilityModel):
    """
    Probability model assuming probabilities follow a normal distribution.

    If combined with MSE similarities, this amounts to assuming a normal
    distribution centered at s_next with a given standard deviation sigma.
    """

    def __init__(self, sigma: float):
        """
        Initialize the Normal Probability Model.

        Parameters
        ----------
        sigma : float
            Standard deviation for the similarity weighting.
            Must be a positive value.

        Raises
        ------
        AssertionError
            If sigma is not a positive value.
        """
        assert sigma > 0, "Sigma must be a positive value"
        self.sigma = sigma

    def unnormalized_log_probability(
        self,
        similarities: NDArray,
        coords_s_next: xr.Dataset,
        coords_candidates: xr.Dataset,
    ) -> NDArray:
        """
        Compute unnormalized log probabilities based on similarities.

        Parameters
        ----------
        similarities : NDArray
            Similarity values for candidate states.
        coords_s_next : xr.Dataset
            Coordinates of the true next state (unused in this model).
        coords_candidates : xr.Dataset
            Coordinates of candidate states (unused in this model).

        Returns
        -------
        NDArray
            Unnormalized log probabilities for each candidate.
        """
        return similarities / (2 * self.sigma**2)


class UniformProbabilityModel(ProbabilityModel):
    """
    Probability model that assigns equal probability to all candidates.

    This model treats all candidate states as equally likely,
    regardless of their similarities or coordinates.
    """

    def __init__(self) -> None:
        """
        Initialize the Uniform Probability Model.

        No parameters are required, as all candidates are treated equally.
        """
        pass

    def unnormalized_log_probability(
        self,
        similarities: NDArray,
        coords_s_next: xr.Dataset,
        coords_candidates: xr.Dataset,
    ) -> NDArray:
        """
        Compute unnormalized log probabilities.

        Returns a constant value of 1 for all candidates,
        effectively making them uniformly probable.

        Parameters
        ----------
        similarities : NDArray
            Similarity values for candidate states (ignored in this model).
        coords_s_next : xr.Dataset
            Coordinates of the true next state (ignored in this model).
        coords_candidates : xr.Dataset
            Coordinates of candidate states (ignored in this model).

        Returns
        -------
        NDArray
            Constant array of ones, representing uniform probabilities.
        """
        return similarities * 0 + 1


class NormalProbabilityModelSeasonality(ProbabilityModel):
    """
    Probability model that incorporates seasonal variability in standard deviation.

    This model adjusts the probability computation based on a sigma that varies
    with the day of the year, reflecting changes in the atmosphere over the year.
    """

    def __init__(
        self, sigma_amplitude: float, normalized_sigma_climatology: xr.DataArray
    ):
        """
        Initialize the Seasonally Variable Probability Model.

        Parameters
        ----------
        sigma_amplitude : float
            Amplitude factor for the climatological sigma.
        normalized_sigma_climatology : xr.DataArray
            Normalized sigma values for each day of the year.

        Notes
        -----
        Sigma is split into amplitude and a normalized climatology to allow rescaling.
        """
        assert (normalized_sigma_climatology > 0).all()
        self.sigma_climatology = normalized_sigma_climatology
        self.sigma_amplitude = sigma_amplitude

    def unnormalized_log_probability(
        self,
        similarities: NDArray,
        coords_s_next: xr.Dataset,
        coords_candidates: xr.Dataset,
    ) -> NDArray:
        """
        Compute unnormalized log probabilities with seasonal sigma adjustment.

        Parameters
        ----------
        similarities : NDArray
            Similarity values for candidate states.
        coords_s_next : xr.Dataset
            Coordinates of the true next state, ignored for this model.
        coords_candidates : xr.Dataset
            Coordinates of candidate states, ignored for this model.

        Returns
        -------
        NDArray
            Unnormalized log probabilities for each candidate.
        """
        return similarities / (
            2
            * (
                self.sigma_climatology.sel(
                    dayofyear=coords_s_next.valid_time.dt.dayofyear
                ).data.item()
                * self.sigma_amplitude
            )
            ** 2
        )


class NormalProbabilityAvoidDirectRepeats(ProbabilityModel):
    """
    Probability model that avoids sampling the base state.

    This model computes probabilities using a Gaussian-like weighting,
    but sets the probability to zero (unnormalized probability negative infinity)
    for candidates that are exact repeats of the base state.
    """

    def __init__(self, sigma: float):
        """
        Initialize the Probability Model that Avoids Direct Repeats.

        Parameters
        ----------
        sigma : float
            Standard deviation for the similarity weighting.
            Must be a positive value.

        Raises
        ------
        AssertionError
            If sigma is not a positive value.
        """
        assert sigma > 0
        self.sigma = sigma

    def unnormalized_log_probability(
        self,
        similarities: NDArray,
        coords_s_next: xr.Dataset,
        coords_candidates: xr.Dataset,
    ) -> NDArray:
        """
        Compute unnormalized log probabilities, excluding direct repeats.

        Parameters
        ----------
        similarities : NDArray
            Similarity values for candidate states.
        coords_s_next : xr.Dataset
            Coordinates of the true next state (unused in this model).
        coords_candidates : xr.Dataset
            Coordinates of candidate states.

        Returns
        -------
        NDArray
            Unnormalized log probabilities for each candidate,
            with direct repeats set to negative infinity.
        """
        mask = (
            (
                coords_s_next.valid_time - np.timedelta64(1, "D")
                == coords_candidates.valid_time
            )
            & (coords_s_next.init_time == coords_candidates.init_time)
            & (coords_s_next.ensemble_member == coords_candidates.ensemble_member)
        )
        res = similarities / (2 * self.sigma**2)

        res[mask] = -np.inf
        return res


class NormalProbabilityKeepMinimalNDays(ProbabilityModel):
    """
    Probability model that enforces a minimum time separation to the true next state.

    This model computes probabilities using a Gaussian-like weighting,
    but sets the probability to zero (unnormalized probability negative infinity) for
    candidates that are within a specified number of days from the true next state.
    """

    def __init__(self, sigma: float, n_days_min: int):
        """
        Initialize the Probability Model with Minimal Time Separation.

        Parameters
        ----------
        sigma : float
            Standard deviation for the similarity weighting.
            Must be a positive value.
        n_days_min : int
            Minimum number of days required between candidate and next state.

        Raises
        ------
        AssertionError
            If sigma is not a positive value.
        """
        assert sigma > 0
        self.sigma = sigma
        self.n_days_min = n_days_min

    def unnormalized_log_probability(
        self,
        similarities: NDArray,  # similarities to other samples
        coords_s_next: xr.Dataset,
        coords_candidates: xr.Dataset,
    ) -> NDArray:
        """
        Compute unnormalized log probabilities, excluding candidates too close in time.

        Parameters
        ----------
        similarities : NDArray
            Similarity values for candidate states.
        coords_s_next : xr.Dataset
            Coordinates of the true next state.
        coords_candidates : xr.Dataset
            Coordinates of candidate states.

        Returns
        -------
        NDArray
            Unnormalized log probabilities for each candidate,
            with candidates too close in time set to negative infinity.
        """
        mask = abs(
            coords_s_next.valid_time - coords_candidates.valid_time
        ) < self.n_days_min * np.timedelta64(1, "D")
        res = similarities / (2 * self.sigma**2)

        res[mask] = -np.inf
        return res


class NormalProbabilityNotLargerThanFixedDate(ProbabilityModel):
    """
    Probability model that restricts candidates to a maximum date.

    This model computes probabilities using a Gaussian-like weighting,
    but sets the probability to zero (unnormalized probability negative infinity)
    for candidates whose date is later than a specified maximum date.
    """

    def __init__(self, sigma: float, date_max: np.datetime64):
        """
        Initialize the Probability Model with Maximum Date Restriction.

        Parameters
        ----------
        sigma : float
            Standard deviation for the similarity weighting.
            Must be a positive value.
        date_max : np.datetime64
            Maximum allowed date for candidate states.

        Raises
        ------
        AssertionError
            If sigma is not a positive value, or if no candidates exist
            before the maximum date.
        """
        assert sigma > 0
        self.sigma = sigma
        self.date_max = date_max

    def unnormalized_log_probability(
        self,
        similarities: NDArray,  # similarities to other samples
        coords_s_next: xr.Dataset,
        coords_candidates: xr.Dataset,
    ) -> NDArray:
        """
        Compute unnormalized log probabilities, excluding candidates after max date.

        Parameters
        ----------
        similarities : NDArray
            Similarity values for candidate states.
        coords_s_next : xr.Dataset
            Coordinates of the true next state (unused in this model).
        coords_candidates : xr.Dataset
            Coordinates of candidate states.

        Returns
        -------
        NDArray
            Unnormalized log probabilities for each candidate,
            with candidates after the maximum date set to negative infinity.

        Raises
        ------
        AssertionError
            If no candidates exist before the maximum date.
        """
        mask = coords_candidates.valid_time > self.date_max
        assert (~mask).any(), (
            f"No candidate found with valid time <= self.date_max = {self.date_max}."
            + " Consider using a more recent starting point."
        )

        res = similarities / (2 * self.sigma**2)
        res[mask] = -np.inf
        return res


# gumbel sampling trick: https://en.wikipedia.org/wiki/Categorical_distribution#Sampling_via_the_Gumbel_distribution
def gumbel_max_sample(
    unnormalized_logp: NDArray,
    rng: np.random.Generator,
    size: int | tuple[int, ...],
) -> NDArray:
    """
    Sample from a categorical distribution using the Gumbel-max trick.

    This method provides an efficient way to sample from a categorical distribution
    by using the properties of the Gumbel distribution.

    Parameters
    ----------
    unnormalized_logp : NDArray
        Unnormalized log probabilities for each category.
    rng : np.random.Generator
        Random number generator.
    size : int or tuple of int
        Number of samples to generate.

    Returns
    -------
    NDArray
        Indices of sampled categories.

    Notes
    -----
    The Gumbel-max trick allows sampling from a categorical distribution
    without explicitly normalizing probabilities.
    """
    if type(size) is int:
        size = (size,)
    z = rng.gumbel(loc=0, scale=1, size=size + unnormalized_logp.shape)
    return (unnormalized_logp + z).reshape(size + (-1,)).argmax(axis=-1)
