# This file is part of swprepost, a Python package for surface wave
# inversion pre- and post-processing.
# Copyright (C) 2019-2020 Joseph P. Vantassel (jvantassel@utexas.edu)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

"""Regular expressions for text parsing."""

import re

NUMBER = r"\d+\.?\d*[eE]?[+-]?\d*"
NEWLINE = r"[\r\n?|\n]"

# DispersionSuite
# ---------------
# Identify the text associated with a single dispersion point.
dc_pair_expr = f"({NUMBER}) ({NUMBER}){NEWLINE}"
dc_pair_expr_no_capture = f"{NUMBER} {NUMBER}{NEWLINE}"
dc_pair_exec = re.compile(dc_pair_expr)

# Identify the text associated with `DispersionSet`.
dc_meta_expr = r"# Layered model (\d+): value=(\d+\.?\d*)"
dc_meta_expr_no_capture = r"# Layered model \d+: value=\d+\.?\d*"
dc_meta_exec = re.compile(dc_meta_expr)

dc_wave_expr = r"# \d+ (Rayleigh|Love) dispersion mode\(s\)"
dc_wave_expr_no_capture = r"# \d+ (?:Rayleigh|Love) dispersion mode\(s\)"
dc_wave_exec = re.compile(dc_wave_expr)

dc_mode_start_expr_a = f"# Mode \d+{NEWLINE}"
dc_mode_start_expr_b = f"# .*{NEWLINE}{dc_meta_expr_no_capture}{NEWLINE}# Mode \d+{NEWLINE}"
dc_mode_start_expr_c = f"{dc_meta_expr_no_capture}{NEWLINE}# Mode \d+{NEWLINE}"
dc_mode_start_exec = re.compile(dc_mode_start_expr_a)

dc_mode_expr = f"# Mode (\d+){NEWLINE}"
dc_mode_exec = re.compile(dc_mode_expr)

# There are three different syntax for dispersion files, dc_header_a, dc_header_b, dc_header_c.
dc_header_a = f"{dc_meta_expr}{NEWLINE}{dc_wave_expr}{NEWLINE}.*{NEWLINE}"
dc_header_b = f"{dc_wave_expr}{NEWLINE}.*{NEWLINE}.*{NEWLINE}{dc_meta_expr}{NEWLINE}"
dc_header_c = f"{dc_wave_expr}{NEWLINE}.*{NEWLINE}{dc_meta_expr}{NEWLINE}"
dc_set_expr = f"(?:{dc_header_a}|{dc_header_b}|{dc_header_c})((?:(?:{dc_mode_start_expr_a}|{dc_mode_start_expr_b}|{dc_mode_start_expr_c})(?:{dc_pair_expr_no_capture})+)+)"
dc_set_exec = re.compile(dc_set_expr)

# GroundModel
# -----------
# Identify the text associated with a single layer of a `GroundModel`.
gm_layer_expr = f"{NUMBER} {NUMBER} {NUMBER} {NUMBER}"
gm_layer_exec = re.compile(f"({NUMBER}) ({NUMBER}) ({NUMBER}) ({NUMBER})")

# Identify the text associated with a single `GroundModel`.
gm_meta_expr = r"# Layered model (\d+): value=(\d+\.?\d*)"
gm_expr = f"{gm_meta_expr}{NEWLINE}\d+{NEWLINE}((?:{gm_layer_expr}{NEWLINE})+)"
gm_exec = re.compile(gm_expr)

# TargetSet
# ---------
# Identify the text associated with a single `ModalCurve`.
modalcurve_expr = r"<ModalCurve>(.*?)</ModalCurve>"
modalcurve_exec = re.compile(modalcurve_expr, re.DOTALL)

# ModalCurveTarget
# -----------
# Given the text associated with a single `ModalCurve` ->
# Find the associated polarization (str).
# Geopsy v2.10.1 uses polarisation, but v3.4.2 uses polarization.
polarization_expr = r"<polari[sz]ation>(Rayleigh|Love)</polari[sz]ation>"
polarization_exec = re.compile(polarization_expr)

# Find the associated Mode (number).
modenumber_expr = r"<index>(\d+)</index>"
modenumber_exec = re.compile(modenumber_expr)

# Find the associated StatPoints (tuple).
statpoint_expr = f"<x>({NUMBER})</x>{NEWLINE}\s*<mean>({NUMBER})</mean>{NEWLINE}\s*<stddev>({NUMBER})</stddev>"
statpoint_exec = re.compile(statpoint_expr)

# Given the text from a swprepost .csv ->
# Find the associated header information.
description_expr = "#(rayleigh|love) (\d+)"
description_exec = re.compile(description_expr)

# Find the associated data
# the first two values (frequency and velocity) are required.
# the third value (velocity standard deviation) is optional.
# TODO(jpv): Deprecate after v2.0.0; remove optionals; require all values.
mtargetpoint_expr = f"({NUMBER}),({NUMBER}),?({NUMBER})?(.*)?{NEWLINE}"
mtargetpoint_exec = re.compile(mtargetpoint_expr)

# This file is part of swprepost, a Python package for surface wave
# inversion pre- and post-processing.
# Copyright (C) 2019-2021 Joseph P. Vantassel (jvantassel@utexas.edu)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

"""DispersionCurve class definition."""

import numpy as np

# from swprepost import Curve, regex
from swprepost import Curve

# __all__ = ['DispersionCurve']


class DispersionCurve(Curve):
    """Class to define a `DispersionCurve` object.

    Attributes
    ----------
    frequency, velocity : ndarray
        1D array of the dispersion curve's frequency and velocity
        values, respectively.

    """

    def __init__(self, frequency, velocity):
        """Initialize a `DispersionCurve` object from dispersion data.

        Parameters
        ----------
        frequency, velocity : iterable
            Vector of the dispersion curve's frequency and velocity
            values, respectively.

        Returns
        -------
        DispersionCurve
            Initialized `DispersionCurve` object.

        """
        super().__init__(x=frequency, y=velocity)

    @property
    def frequency(self):
        return self._x

    @property
    def velocity(self):
        return self._y

    @property
    def wavelength(self):
        return self._y/self._x

    @property
    def slowness(self):
        return 1/self._y

    @classmethod
    def _parse_dc(cls, dc_data):
        """Parse a single `DispersionCurve` from dispersion data.

        Parameters
        ----------
        dc_data : str
            Dispersion curve data of the form `frequency, slowness`.
            It is assumed that frequencies increases monotonically. If
            this assumption is not true, incorrect results will result.
            See example below.

        Returns
        -------
        DispersionCurve
            Instantiated `DispersionCurve` object.

        Example
        -------
        If `dc_data` is as follows:
            Line 1: # Frequency, Slowness
            Line 2: 0.1, 0.01
            Line 3: 0.2, 0.012
            Line 4: # Frequency, Slowness
            Line 5: 0.1, 0.011
            Line 6: 0.2, 0.013
        Only lines 2 and 3 will be parsed.

        """
        frequency, slowness = [], []
        for curve in dc_pair_exec.finditer(dc_data):
            f, p = curve.groups()
            f = float(f)
            try:
                if f < frequency[-1]:
                    break
                else:
                    frequency.append(f)
                    slowness.append(float(p))
            except IndexError:
                frequency.append(f)
                slowness.append(float(p))
        return cls(frequency=frequency, velocity=1/np.array(slowness,
                                                            dtype=np.double))

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
            lines = f.read()
        return cls._parse_dc(lines)

    @property
    def txt_repr(self):
        """Text representation following the Geopsy format."""
        lines = ""
        for f, p in zip(self.frequency, self.slowness):
            lines += f"{f} {p}\n"
        return lines

    def write_curve(self, fileobj):
        """Append `DispersionCurve` to open file object.

        Parameters
        ----------
        fname : str
            Name of file, may be a relative or the full path.

        Returns
        -------
        None
            Writes file to disk.

        """
        fileobj.write(self.txt_repr)

    def write_to_txt(self, fname, wavetype="rayleigh", mode=0,
                     identifier=0, misfit=0.0000):
        """Write `DispersionCurve` to Geopsy formated file.

        Parameters
        ----------
        fname : str
            Name of file, may be a relative or the full path.
        wavetype : {"rayleigh", "love"}, optional
            Surface wave dispersion wavetype, default is "rayleigh".
        mode : int, optional
            Mode integer (numbered from zero), default is 0.
        identifier : int, optional
            Model identifier, default is 0.
        misfit : float, optional
            Dispersion misfit of profile, default is 0.0000.

        Returns
        -------
        None
            Write text representation to disk.

        """
        with open(fname, "w") as f:
            f.write("# File written by swprepost\n")
            f.write(f"# Layered model {identifier}: value={misfit}\n")
            f.write(f"# 1 {wavetype.capitalize()} dispersion mode(s)\n")
            f.write("# CPU Time = 0 ms\n")
            f.write(f"# Mode {mode}\n")
            self.write_curve(f)

    def __eq__(self, other):
        """Define when two `GroundModel` are equal."""
        for attr in ["frequency", "velocity"]:
            my_vals = getattr(self, attr)
            ur_vals = getattr(other, attr)
            if len(my_vals) != len(ur_vals):
                return False
            for my, ur in zip(my_vals, ur_vals):
                if np.round(my, 6) != np.round(ur, 6):
                    return False
        return True

    def __repr__(self):
        """Unambiguous representation of a `DispersionCurve` object."""
        return f"DispersionCurve(frequency={self.frequency}, velocity={self.velocity})"

    def __str__(self):
        """Readable representation of a `DispersionCurve` object."""
        return f"DispersionCurve with {len(self.frequency)} points"

# This file is part of swprepost, a Python package for surface wave
# inversion pre- and post-processing.
# Copyright (C) 2019-2020 Joseph P. Vantassel (jvantassel@utexas.edu)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

"""DispersionSet class definition."""

import numpy as np

# from swprepost import DispersionCurve, regex

# __all__ = ["DispersionSet"]


class DispersionSet():
    """Class for handling sets of
    :meth: `DispersionCurve <swprepost.DispersionCurve>` objects, which all
    belong to a common ground model.

    Attributes
    ----------
    rayleigh, love : dict
        Container for `DispersionCurve` objects, of the form:
        `{0:DispersionCurve0, ... N:DispersionCurveN}`
        where each key is the mode number and the value is the
        corresponding instantiated `DispersionCurve` object.
    identifier : int
        Model identifier of the `DispersionSet`.
    misfit : float
        Value of dispersion misfit if provided, `None` otherwise.

    """

    @classmethod
    def check_type(cls, curveset, valid_type):
        """Check that the `curveset` are are valid.

        Specifically:
        1. Assume `curveset` is instance of `dict`.
        2. If it is a `dict`, check all values are instances of the
        `valid_type` and return zero, otherwise raise `TypeError`.
        3. If it is not check if `None`, if so return one.
        4. Otherwise, raise `TypeError`.

        """
        try:
            for key, value in curveset.items():
                if not isinstance(value, valid_type):
                    msg = f"{key} must be a {valid_type}, not {type(value)}."
                    raise TypeError(msg)
        except AttributeError:
            if curveset is None:
                return 1
            else:
                msg = f"CurveSet must be a `dict` or `None`, not {type(curveset)}."
                raise TypeError(msg)
        return 0

    def __init__(self, identifier=0, misfit=0.0, rayleigh=None, love=None):
        """Create a `DispersionCurveSet` object.

        Parameters
        ----------
        identifier : str
            Unique identifier of the `DispersionSet`.
        misfit : float, optional
            `DispersionSet` misfit, default is 0.0.
        rayleigh, love : dict
            Container for `DispersionCurve` objects of the form
            `{0:disp_curve_obj0, ... N:disp_curve_objN}` where each
            key is the mode number and the value is the
            corresponding `DispersionCurve` object.

        Returns
        -------
        DispersionSet
            Instantiated `DispersionSet` object.

        """
        none_count = 0
        none_count += self.check_type(rayleigh, self._dc())
        none_count += self.check_type(love, self._dc())

        if none_count == 2:
            msg = "`rayleigh` and `love` cannot both be `None`."
            raise ValueError(msg)

        self.rayleigh = None if rayleigh is None else dict(rayleigh)
        self.love = None if love is None else dict(love)

        self.identifier = int(identifier)
        self.misfit = float(misfit)

    @classmethod
    def _parse_dcs(cls, dcs_data, nmodes="all"):
        """Parse a group of modes into a `dict` of `DispersionCurves`"""
        modes = dc_mode_start_exec.split(dcs_data)

        if nmodes == "all":
            modes = modes[1:]
        elif nmodes == 0:
            return None
        else:
            modes = modes[1:nmodes+1]

        dcs = {}
        for mode, dc_data in zip(dc_mode_exec.finditer(dcs_data), modes):
            mode_number = int(mode.groups()[0])
            dcs.update({mode_number: cls._dc()._parse_dc(dc_data)})
        return dcs

    @classmethod
    def _from_full_file(cls, text, nrayleigh="all", nlove="all"):
        """Parse the first `DispersionSet` from Geopsy-style contents.

        Parameters
        ----------
        text : str
            Contents of Geopsy-style text file.
        nrayleigh, nlove : {"all", int}, optional
            Number of Rayleigh and Love modes to extract into a
            `DispersionSet` object, default is "all" meaning all
            available modes will be extracted.

        Returns
        -------
        DispersionSet
            Instantiated `DispersionSet` object.

        """
        if nrayleigh == 0 and nlove == 0:
            raise ValueError(f"`nrayleigh` and `nlove` cannot both be 0.")

        rayleigh, love = None, None
        previous_id, previous_misfit = "start", "0"
        for model_info in dc_set_exec.finditer(text):
            id_a, msft_a, wav_a, wav_b, id_b, msft_b, wav_c, id_c, msft_c,  data = model_info.groups()

            for _id, _msft, _wav in zip([id_a, id_b, id_c], [msft_a, msft_b, msft_c], [wav_a, wav_b, wav_c]):
                if _id is not None:
                    identifier = _id
                    misfit = _msft
                    wave_type = _wav
                    break

            if identifier == previous_id or previous_id == "start":
                if wave_type == "Rayleigh":
                    rayleigh = cls._parse_dcs(data, nmodes=nrayleigh)
                elif wave_type == "Love":
                    love = cls._parse_dcs(data, nmodes=nlove)
                else: # pragma: no cover
                    raise NotImplementedError
                previous_id = identifier
                previous_misfit = misfit
            else:
                break

        return cls(previous_id, float(previous_misfit),
                   rayleigh=rayleigh, love=love)

    @classmethod
    def _dc(cls):
        """Define `DispersionCurve` to allow subclassing."""
        return DispersionCurve

    @classmethod
    def from_geopsy(cls, fname, nrayleigh="all", nlove="all"):
        """Create from a text file following the Geopsy format.

        Parameters
        ----------
        fname : str
            Name of file to be read, may be a relative or full path.
        nrayleigh, nlove : {"all", int}, optional
            Number of Rayleigh and Love modes to extract into a
            `DispersionSet` object, default is "all" meaning all
            available modes will be extracted.

        Returns
        -------
        DispersionSet
            Instantiated `DispersionSet` object.

        """
        with open(fname, "r") as f:
            text = f.read()
        return cls._from_full_file(text, nrayleigh=nrayleigh, nlove=nlove)

    def write_set(self, fileobj, nrayleigh="all", nlove="all"):
        """Write `DispersionSet` to current file.

        Parameters
        ----------
        fname : str
            Name of file, may be a relative or the full path.

        Returns
        -------
        None
            Writes file to disk.

        """
        nrayleigh = np.inf if nrayleigh == "all" else int(nrayleigh)
        nlove = np.inf if nlove == "all" else int(nlove)

        misfit = 0.0 if self.misfit is None else self.misfit
        if (self.rayleigh is not None) and (nrayleigh > 0):
            fileobj.write(
                f"# Layered model {self.identifier}: value={misfit}\n")
            nmodes = min(len(self.rayleigh), nrayleigh)
            # TODO (jpv): Not true is mode is missing.
            fileobj.write(f"# {nmodes} Rayleigh dispersion mode(s)\n")
            fileobj.write("# CPU Time = 0 ms\n")
            for key, value in self.rayleigh.items():
                if key >= nrayleigh:
                    continue
                fileobj.write(f"# Mode {key}\n")
                value.write_curve(fileobj)
        if (self.love is not None) and (nlove > 0):
            fileobj.write(
                f"# Layered model {self.identifier}: value={misfit}\n")
            nmodes = min(len(self.love), nlove)
            # TODO (jpv): Not true is mode is missing.
            fileobj.write(f"# {nmodes} Love dispersion mode(s)\n")
            fileobj.write("# CPU Time = 0 ms\n")
            for key, value in self.love.items():
                if key >= nlove:
                    continue
                fileobj.write(f"# Mode {key}\n")
                value.write_curve(fileobj)

    def write_to_txt(self, fname):
        """Write `DispersionSet` to Geopsy formated file.

        Parameters
        ----------
        fname : str
            Name of file, may be a relative or the full path.

        Returns
        -------
        None
            Writes text representation to disk.

        """
        with open(fname, "w") as f:
            f.write("# File written by swprepost\n")
            self.write_set(f)

    def __eq__(self, other):
        """Define when two `DispersionSet` objects are equal."""
        for attr in ["misfit", "identifier", "love", "rayleigh"]:
            my_attr = getattr(self, attr)
            ur_attr = getattr(other, attr)
            if my_attr != ur_attr:
                return False
        return True

    def __repr__(self):
        """Unambiguous representation of a `DispersionSet` object."""
        return f"DispersionSet(identifier={self.identifier}, rayleigh={self.rayleigh}, love={self.love}, misfit={self.misfit})"

    def __str__(self):
        """Human-readable representation of `DispersionSet` object."""
        return f"DispersionSet with {len(self.rayleigh)} Rayleigh and {len(self.love)} Love modes"

# This file is part of swprepost, a Python package for surface wave
# inversion pre- and post-processing.
# Copyright (C) 2019-2020 Joseph P. Vantassel (jvantassel@utexas.edu)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

"""DispersionSuite class definition."""

import logging
import warnings

# import numpy as np

# from swprepost import DispersionSet, Suite, regex
from swprepost import Suite

logger = logging.getLogger(__name__)

# __all__ = ["DispersionSuite"]


class DispersionSuite(Suite):
    """Container for instantiated `DispersionSet` objects.

    Attributes
    ----------
    sets : list
        Container for instantiated `DispersionSet` objects.

    """
    @staticmethod
    def check_input(curveset, set_type):
        """Check inputs comply with the required format.

        Specifically:
        1. `curveset` is of type `set_type`.

        """
        if not isinstance(curveset, set_type):
            msg = f"Must be instance of {type(set_type)}, not {type(curveset)}."
            raise TypeError(msg)

    def __init__(self, dispersionset):
        """Initialize a `DispersionSuite`, from a `DispersionSet`.

        Parameters
        ----------
        dispersionset : DispersionSet
            Initialized `DispersionSet` object.

        Returns
        -------
        DispersionSuite
            Instantiated `DispersionSuite` object.

        Raises
        ------
        TypeError
            If `dispersionset` is not of type `DispersionSet`.

        """
        self.check_input(dispersionset, DispersionSet)
        super().__init__(dispersionset)

    @property
    def sets(self):
        return self._items

    def append(self, dispersionset, sort=True):
        """Append `DispersionSet` object to `DispersionSuite`.

        Parameters
        ----------
            Refer to :meth: `__init__ <DispersionSuite.__init__>`.

        Returns
        -------
        None
            Updates the attribute `sets`.

        Raises
        ------
        TypeError
            If `dispersionset` is not of type `DispersionSet`.

        """
        self.check_input(dispersionset, DispersionSet)
        super()._append(dispersionset, sort=sort)

    @classmethod
    def from_geopsy(cls, fname, nsets="all", nrayleigh="all", nlove="all",
                    sort=False):
        """Instantiate from a text file following the Geopsy format.

        Parameters
        ----------
        fname : str
            Name of file, may be a relative or full path.
        nsets : int, optional
            Number of sets to extract, default is "all" so all
            available sets will be extracted.
        nrayleigh, nlove : int, optional
            Number of Rayleigh and Love modes respectively, default
            is "all" so all available modes will be extracted.
        sort : bool, optional
            Indicates whether the imported data should be sorted from
            lowest to highest misfit, default is `False` indicating no
            sorting is performed.

        Returns
        -------
        DispersionSuite
            Instantiated `DispersionSuite` object.

        """
        if nsets == "all":
            nsets = np.inf

        with open(fname, "r") as f:
            text = f.read()

        dc_sets = []
        previous_id, previous_misfit = "start", "0"
        rayleigh, love = None, None
        model_count = 0
        for model_info in dc_set_exec.finditer(text):
            id_a, msft_a, wav_a, wav_b, id_b, msft_b, wav_c, id_c, msft_c,  data = model_info.groups()

            for _id, _msft, _wav in zip([id_a, id_b, id_c], [msft_a, msft_b, msft_c], [wav_a, wav_b, wav_c]):
                if _id is not None:
                    identifier = _id
                    misfit = _msft
                    wave_type = _wav
                    break

            # Encountered new model, save previous, and reset.
            if identifier != previous_id and previous_id != "start":
                if model_count+1 == nsets:
                    break

                dc_sets.append(cls._dcset()(previous_id,
                                            float(previous_misfit),
                                            rayleigh=rayleigh, love=love))
                model_count += 1
                rayleigh, love = None, None

            # Parse data.
            if wave_type == "Rayleigh":
                rayleigh = cls._dcset()._parse_dcs(data, nmodes=nrayleigh)
            elif wave_type == "Love":
                love = cls._dcset()._parse_dcs(data, nmodes=nlove)
            else: # pragma: no cover
                raise NotImplementedError

            previous_id, previous_misfit = identifier, misfit

        dc_sets.append(cls._dcset()(previous_id,
                                    float(previous_misfit),
                                    rayleigh=rayleigh, love=love))

        if nsets is not np.inf and len(dc_sets) < nsets:
            msg =  f"The number of DispersionSets requested ({nsets}) is "
            msg += f"fewer than the number of those returned ({len(dc_sets)})."
            warnings.warn(msg, UserWarning)

        return cls.from_list(dc_sets, sort=sort)

    @classmethod
    def _dcset(cls):
        """Convenient `DispersionSet` to allow subclassing."""
        return DispersionSet

    @classmethod
    def from_list(cls, dc_sets, sort=True):
        """Instantiate from a list of `DispersionSet` objects.

        Parameters
        ----------
        dc_sets : list
            List of `DispersionSet` objects.
        sort : bool, optional
            Indicates whether the imported data should be sorted from
            lowest to highest misfit, default is `False` indicating no
            sorting is performed.

        Returns
        -------
        DipsersionSuite
            Instantiated `DispersionSuite` object.

        """
        obj = cls(dc_sets[0])
        if len(dc_sets) > 1:
            for dc_set in dc_sets[1:]:
                obj.append(dc_set, sort=sort)
        return obj

    def write_to_txt(self, fname, nbest="all", nrayleigh="all", nlove="all"):
        """Write to text file, following the Geopsy format.

        Parameters
        ----------
        fname : str
            Name of file, may be a relative or the full path.
        nbest : {int, 'all'}, optional
            Number of best models to write to file, default is 'all'
            indicating all models will be written.
        nrayleigh, nlove : {int, 'all'}, optional
            Number of modes to write to file, default is 'all'
            indicating all available modes will be written.

        Returns
        -------
        None
            Writes file to disk.

        """
        nbest = self._handle_nbest(nbest)
        with open(fname, "w") as f:
            f.write("# File written by swprepost\n")
            for cit in self.sets[:nbest]:
                cit.write_set(f, nrayleigh=nrayleigh, nlove=nlove)

    def __getitem__(self, slce):
        """Define slicing behavior"""
        return self.sets[slce]

    def __str__(self):
        """Human-readable representation of the object."""
        return f"DispersionSuite with {len(self.sets)} DispersionSets."
