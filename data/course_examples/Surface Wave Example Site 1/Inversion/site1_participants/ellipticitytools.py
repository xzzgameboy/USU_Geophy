# This file is part of swprepostpp, a Python package for surface wave
# inversion pre- and post-processing.
# Copyright (C) 2019-2023 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)

import re

NUMBER = r"\d+\.?\d*[eE]?[+-]?\d*"
NEWLINE = r"[\r\n?|\n]"

# EllipticitySuite
# ---------------
# identify number of rayleigh ellipticity modes.
ell_wave_expr = r"# (\d+) Rayleigh ellipticity mode\(s\)"
ell_wave_exec = re.compile(ell_wave_expr)

# identify model and misfit associated with ellipticity.
ell_model_and_misfit_expr = r"# Layered model (\d+): value=(\d+\.?\d*)"
ell_model_and_misfit_exec = re.compile(ell_model_and_misfit_expr)

# identify mode associated with ellipticity.
ell_mode_expr = f"# Mode (\d+)"
ell_mode_exec = re.compile(ell_mode_expr)

# identify start of a new ellipticity set (possibility a - start of file).
ell_set_start_a_expr = ""
ell_set_start_a_expr += f"{ell_wave_expr}{NEWLINE}"
ell_set_start_a_expr += f"# \d+ .+{NEWLINE}"
ell_set_start_a_expr += f"{ell_model_and_misfit_expr}{NEWLINE}"
ell_set_start_a_expr += f"{ell_mode_expr}{NEWLINE}"
ell_set_start_a_exec = re.compile(ell_set_start_a_expr)

# identify start of a new ellipticity set (possibility b - otherwise).
ell_set_start_b_expr = ""
ell_set_start_b_expr += f"{ell_wave_expr}{NEWLINE}"
ell_set_start_b_expr += f"{ell_model_and_misfit_expr}{NEWLINE}"
ell_set_start_b_expr += f"{ell_mode_expr}{NEWLINE}"
ell_set_start_b_exec = re.compile(ell_set_start_b_expr)

# identify next mode in ellipticity set (possibility a - start of file).
ell_next_mode_start_a_expr = ""
ell_next_mode_start_a_expr += f"# \d\d+ [^R].+{NEWLINE}"
ell_next_mode_start_a_expr += f"{ell_model_and_misfit_expr}{NEWLINE}"
ell_next_mode_start_a_expr += f"{ell_mode_expr}{NEWLINE}"
ell_next_mode_start_a_exec = re.compile(ell_next_mode_start_a_expr)

# identify next mode in ellipticity set (possibility b - otherwise).
ell_next_mode_start_b_expr = ""
ell_next_mode_start_b_expr += f"{ell_model_and_misfit_expr}{NEWLINE}"
ell_next_mode_start_b_expr += f"{ell_mode_expr}{NEWLINE}"
ell_next_mode_start_b_exec = re.compile(ell_next_mode_start_b_expr)

# identify a single ellipticity point.
ell_pair_expr = f"({NUMBER}) (-?{NUMBER})"
ell_pair_exec = re.compile(ell_pair_expr)

# This file is part of swprepostpp, a Python package for surface wave
# inversion pre- and post-processing.
# Copyright (C) 2019-2023 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)

# """EllipticityCurve class definition."""

import numpy as np
from swprepost import Curve

# from .regex import ell_set_start_a_exec, ell_set_start_b_exec, ell_next_mode_start_a_exec, ell_pair_exec


class EllipticityCurve(Curve):
    """Ellipticity for a particular Rayleigh mode and specific model.

    Attributes
    ----------
    frequency : array-like
        Denotes the ellipticity curve's frequency values.
    ellipticity : array-like
        Denotes the ellipticity curve's ellipticity values
        (one per frequency).

    """

    def __init__(self, frequency, ellipticity):
        """Initialize an `EllipticityCurve`.

        Parameters
        ----------
        frequency : array-like
            Denotes the ellipticity curve's frequency values.
        ellipticity : array-like
            Denotes the ellipticity curve's ellipticity values
            (one per frequency).

        Returns
        -------
        EllipticityCurve
            Instantiated `EllipticityCurve` object.

        """
        super().__init__(x=frequency, y=ellipticity)

    @property
    def frequency(self):
        return self._x

    @property
    def ellipticity(self):
        return self._y

    @property
    def ellipticity_signed(self):
        return self._y

    @property
    def ellipticity_absolute(self):
        return np.abs(self._y)

    @property
    def ellipticity_angle_radian(self):
        return np.arctan(self._y)
        # return np.arctan2(self._y)

    @property
    def ellipticity_angle_degree(self):
        return np.rad2deg(np.arctan(self._y))
        # return np.rad2deg(np.arctan2(self._y))

    @classmethod
    def _parse_ell(cls, text):
        """Parse a single `EllipticityCurve` from ellipticity data.

        .. warning::
            Private API is subject to change without warning.

        Parameters
        ----------
        text : str
            Text in the Geopsy format containing one a single mode of
            ellipticity data.

        Returns
        -------
        EllipticityCurve
            Instantiated ``EllipticityCurve`` object.

        Example
        -------
        ``text`` is assumed to appear as follows:            
            Line 1:  0.845 -1.003
            Line 2:  0.919 -1.044

        """
        frequency, ellipticity = [], []
        for match in ell_pair_exec.finditer(text):
            f, e = match.groups()
            f, e = float(f), float(e)
            frequency.append(f)
            ellipticity.append(e)
        return cls(frequency=frequency, ellipticity=ellipticity)

    @classmethod
    def from_geopsy(cls, fname):
        """Create from text file following the Geopsy format.

        Parameters
        ----------
        fname : str
            Name of file to be read, may be a relative or the full path.

        Returns
        -------
        DispersionCurve
            Instantiated `DispersionCurve` object.

        """
        with open(fname, "r") as f:
            text = f.read()

        match = ell_set_start_a_exec.search(text)
        nmodes, *_ = match.groups()
        start_idx = match.end()

        if nmodes != "1":
            dx_match = ell_next_mode_start_a_exec.search(text[start_idx:])
        else:
            dx_match = ell_set_start_b_exec.search(text[start_idx:])
        end_idx = len(text) if dx_match is None else start_idx + dx_match.start()

        return cls._parse_ell(text[start_idx:end_idx])

    @classmethod
    def from_ellipticitycurve(cls, ellipticitycurve):
        """Copy constructor for `EllipticityCurve`.

        Parameters
        ----------
        ellipticitycurve : EllipticityCurve
            Curve to be copied.

        Returns
        -------
        EllipticityCurve
            Copy of provided ellipticitycurve.

        """
        return cls(ellipticitycurve.frequency, ellipticitycurve.ellipticity)

# This file is part of swprepostpp, a Python package for surface wave
# inversion pre- and post-processing.
# Copyright (C) 2019-2023 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)

# """EllipticitySet class definition."""

# import numpy as np

# from .ellipticitycurve import EllipticityCurve
# from .regex import ell_next_mode_start_a_exec, ell_next_mode_start_b_exec, ell_set_start_a_exec, ell_set_start_b_exec

class EllipticitySet():
    """Class for handling sets of EllipticityCurve objects which all
    belong to a common velocity model.

    Attributes
    ----------
    ellipticity : dict
        Of the form `{0:EllipticityCurve0,..., N:EllipticityCurveN}`
        where each key represents the Rayleigh mode number and each
        value the corresponding `EllipticityCurve` object.

    """

    def __init__(self, ellipticityset, identifier=0, misfit=0.):
        """Initializes an `EllipticitySet`.

        Parameters
        ----------
        ellipticity : dict
            Of the form `{0:EllipticityCurve0,..., N:EllipticityCurveN}`
            where each key represents the Rayleigh mode number and each
            value the corresponding `EllipticityCurve` object.
        identifier : int, optional
            Unique identifier of the `EllipticitySet`, default is 0.
        misfit : float, optional
            `EllipticitySet` misfit, default is 0.0.

        Returns
        -------
        EllipticitySet
            Initialized `EllipticitySet`.

        """
        self.ellipticity = dict(ellipticityset)
        self.identifier = int(identifier)
        self.misfit = float(misfit)

    @classmethod
    def _parse_ell_set(cls, text, nmodes="all"):
        """Parse text into a `dict` of `EllipticityCurves`
        
        .. warning::
            Private API is subject to change without warning.

        Parameters
        ----------
        text : str
            Text in the Geopsy format containing data for one
            ellipticity set.
        nmodes : {int, str}, optional
            Number of ellipticity modes to extract. Default is
            ``"all"`` which will extract all available modes.

        Returns
        -------
        EllipticityCurve
            Instantiated ``EllipticityCurve`` object.

        """
        nmodes = np.inf if nmodes == "all" else nmodes

        # handle first vs all other sets
        if ell_next_mode_start_a_exec.search(text) is not None:
            regex_exec = ell_next_mode_start_a_exec
        else:
            regex_exec = ell_next_mode_start_b_exec

        models = []
        misfits = []
        modes = []
        start_idxs, end_idxs = [], []
        for match in regex_exec.finditer(text):
            end_idxs.append(match.start())
            start_idxs.append(match.end())
            
            # metadata
            model, misfit, mode = match.groups()
            models.append(int(model))
            misfits.append(float(misfit))
            modes.append(int(mode))

            if len(start_idxs) == nmodes + 1:
                break
        else:
            start_idxs.append(-1)
            end_idxs.append(-1)
        start_idxs = start_idxs[:-1]
        end_idxs = end_idxs[1:]

        # check model and misfit
        if len(set(misfits)) != 1 or len(set(models)) != 1:
            msg = "Should only be one misfit and one model per "
            msg += "ellipticity set."
            raise ValueError(msg)

        ell_set = {}
        for start_idx, end_idx, mode in zip(start_idxs, end_idxs, modes):
            ell_set[int(mode)] = EllipticityCurve._parse_ell(text[start_idx:end_idx])
        
        return cls(ell_set, identifier=models[0], misfit=misfits[0])
    
    @classmethod
    def from_geopsy(cls, fname, nmodes="all"):
        """Create from text file following the Geopsy format.

        Parameters
        ----------
        fname : str
            Name of file to be read, may be a relative or the full path.
        nmodes : {int, str}, optional
            Number of ellipticity modes to extract. Default is
            ``"all"`` which will extract all available modes.

        Returns
        -------
        DispersionCurve
            Instantiated `DispersionCurve` object.

        """
        with open(fname, "r") as f:
            text = f.read()

        start_match = ell_set_start_a_exec.search(text)
        start_idx = start_match.start()
        dx_match = ell_set_start_b_exec.search(text[start_match.end():])
        end_idx = -1 if dx_match is None else start_match.end() + dx_match.start()

        return cls._parse_ell_set(text[start_idx:end_idx], nmodes=nmodes)        

    def __getitem__(self, sliced):
        return self.ellipticity[sliced]

# # This file is part of swprepostpp, a Python package for surface wave
# inversion pre- and post-processing.
# Copyright (C) 2019-2023 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)


"""Definition for Suite."""

import logging

from swprepost import Suite as SuiteBasic

logger = logging.getLogger(name=__name__)

class Suite(SuiteBasic):

    def __init__(self, item):
        super().__init__(item)
    
    @classmethod
    def from_suites(cls, suites, sort=True):
        """Create from iterable of `Suite` objects."""
        items, misfits = [], []
        for suite in suites:
            items += suite._items
            misfits += suite.misfits

        if sort:
            items = [item for item, _ in sorted(zip(items, misfits), key=lambda x: x[1])]
        del misfits

        obj = cls(items[0])
        for item in items[1:]:
            obj.append(item, sort=False)
        return obj
    
    def __getitem__(self, sliced):
        return self._items[sliced]

# This file is part of swprepostpp, a Python package for surface wave
# inversion pre- and post-processing.
# Copyright (C) 2019-2023 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)

# """EllipticitySuite class definition."""

import warnings

# import numpy as np

# from .ellipticitycurve import EllipticityCurve
# from .ellipticityset import EllipticitySet
# from .suite import Suite
# from .regex import ell_set_start_a_exec, ell_set_start_b_exec
# from .meta import SUPPORTED_GEOPSY_VERSIONS
from swprepost.meta import SUPPORTED_GEOPSY_VERSIONS

class EllipticitySuite(Suite):
    """Suite of `EllipticitySet` objects."""

    def __init__(self, ellipticityset):
        """Initialize an `EllipticitySuite` object.

        Parameters
        ----------
        ellipticityset : EllipticitySet
            Instantiated `EllipticitySet` object.

        Returns
        -------
        EllipticitySuite
            Instantiated `EllipticitySuite` object.

        """
        super().__init__(item=ellipticityset)

    def append(self, ellipticityset, sort=True):
        """Append `EllipticitySet` to `EllipticitySuite`.

        Parameters
        ----------
        ellipticityset : EllipticitySet
            Instantiated `EllipticitySet` object.

        Returns
        -------
        None
            This method returns no value, instead updates the state of
            the object upon which it was called.

        """
        super()._append(ellipticityset, sort=sort)

    @property
    def sets(self):
        return self._items

    @classmethod
    def from_geopsy(cls, fname, nsets="all", nmodes="all", sort=False):
        """Create ``EllipticitySuite`` from a text file created by ``gpell``.

        Parameters
        ----------
        fname : str
            File name, may contain a relative or the full path.
        nsets : int
            Number of ``EllipticitySet`` to extract. Default is
            ``"all"`` which will extract all available sets.
        nmodes : int
            Number of ellipticity modes to extract. Default is
            ``"all"`` which will extract all available modes.
        sort : bool, optional
            Indicates whether the imported data should be sorted from
            lowest to highest misfit, default is ``False`` indicating no
            sorting is performed.

        Returns
        -------
        EllipticitySuite
            Instantiated `EllipticitySuite` class.

        """
        nsets = np.inf if nsets == "all" else int(nsets)

        with open(fname, "r") as f:
            text = f.read()

        # identify the text associated with each ellipticity set.
        start_idxs = []
        end_idxs = []

        # first set is unique
        match = ell_set_start_a_exec.search(text)
        if match is None:
            msg = "Incorrect file format, check Geopsy version is in "
            msg += "the following list of support versions: "
            msg += f"{SUPPORTED_GEOPSY_VERSIONS}"
            raise ValueError(msg)
        start_idxs.append(match.start())

        # all other sets
        for match in ell_set_start_b_exec.finditer(text):
            end_idxs.append(match.start())
            if len(start_idxs) == nsets:
                break
            start_idxs.append(match.start())
        else:
            end_idxs.append(len(text))
            # if ~np.isinf(nsets):
            #     msg = f"The number of sets requested {nsets} exceeds the "
            #     msg += f"the number of sets available {len(start_idxs)}."
            #     warnings.warn(msg)

        # parse sets
        ell_sets = []
        for start_idx, end_idx in zip(start_idxs, end_idxs):
            ell_sets.append(EllipticitySet._parse_ell_set(text[start_idx:end_idx],
                                                          nmodes=nmodes))
        return cls.from_list(ell_sets, sort=sort)

    @classmethod
    def from_list(cls, ell_sets, sort=True):
        """Instantiate from a list of `EllipticitySet` objects.

        Parameters
        ----------
        ell_sets : list
            List of `EllipticitySet` objects.
        sort : bool, optional
            Indicates whether the imported data should be sorted from
            lowest to highest misfit, default is `False` indicating no
            sorting is performed.

        Returns
        -------
        EllipticitySuite
            Instantiated `EllipticitySuite` object.

        """
        obj = cls(ell_sets[0])
        if len(ell_sets) > 1:
            for ell_set in ell_sets[1:]:
                obj.append(ell_set, sort=sort)
        return obj

    @classmethod
    def from_suites(cls, suites, sort=True):
        """Create new suite from a list of `EllipticitySuites`."""
        obj = cls(suites[0].sets[0])
        starting_index = 1
        for suite in suites:
            for cset in suite.sets[starting_index:]:
                obj.append(cset, sort=False)
            starting_index = 0

        if sort:
            obj._sort()

        return obj

    def plot(self, ax, nbest="all", nell=1, **kwargs):
        """Plot `EllipticitySuite`."""
        # TODO (jpv): Write docstring.
        nbest = self._handle_nbest(nbest)

        if nell < 1:
            raise ValueError("nell must be > 0.")

        for nset in range(nbest):
            for cell in range(nell):
                ax.plot(self.sets[nset].ellipticity[cell].frequency,
                        self.sets[nset].ellipticity[cell].ellipticity,
                        **kwargs)
                kwargs['label'] = None

    def __str__(self):
        """Human-readable representation of the object."""
        return f"EllipticitySuite with {len(self.sets)} EllipticitySets."

