from itertools import chain, combinations
import os
from pathlib import Path
from typing import Sequence, Tuple
from string import ascii_uppercase

from hikari.dataframes import HklFrame
from hikari.symmetry import SG
import numpy as np
import pandas as pd

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INPUT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

unit_cell = dict(a=9.37, b=9.37, c=6.88, al=90.0, be=90.0, ga=120.0)
space_group = 'P63/m'
opening_angle = 55.0
wavelength = 0.41
d_min = 0.5
reciprocal_space_direction_perp_to_dac = [
    # use [brackets] for reciprocal space vectors i.e. X = [1.0, 0.0, 0.0]
    # or (parentheses) for face miller indices i.e. face (1, 0, 0)
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    (1, 0, 1)
]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

max_hkl_frame_count = 10
pd.options.display.max_columns = None
pd.options.display.width = None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


resolution = min([2/wavelength, 1/d_min])


class HklStats:
    """Class calculating statistics for each hkl data frame"""
    def __init__(self, full_hkl_frame: HklFrame) -> 'HklStats':
        columns = ['total', 'unique', 'completeness', 'redundancy']
        self.table = pd.DataFrame(columns=columns)
        self.table.loc['full'] = self.counts(full_hkl_frame)

    def counts(self, hkl_frame) -> Tuple[int, int, float, float]:
        """Count statistics for `hkl_frame` and return them as a tuple"""
        total = len(hkl_frame.table)
        unique = hkl_frame.table['equiv'].nunique()
        completeness = (unique / self.table.loc['full', 'unique']) \
            if 'full' in self.table.index else 1.0
        redundancy = total / unique
        return total, unique, completeness, redundancy

    def register(self, name: str, hkl_frame: HklFrame) -> None:
        """Register statistics of a new `hkl_frame` in self under a `name`"""
        self.table.loc[name] = self.counts(hkl_frame)
        self.table['total'] = self.table['total'].astype(int)
        self.table['unique'] = self.table['unique'].astype(int)


def add_hkl_frames(hkl_frames: Sequence[HklFrame]) -> HklFrame:
    """Return a `HklFrame` representing a sum of supplied `hkl_frames`"""
    assert len(hkl_frames) > 0
    hkl_sum = hkl_frames[0].copy()
    for hkl_frame in hkl_frames[1:]:
        hkl_sum += hkl_frame
    hkl_sum.name = ''.join(h.name for h in hkl_frames)  # noqa
    return hkl_sum


def powerset(iterable):
    """powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


def save_nice_hkl_res(hkl_frame: HklFrame, name: str) -> None:
    """Modify the hkl frame multiplicity to improve coloring and save file"""
    subdir = Path(__file__).parent / 'hkl'
    if not os.path.isdir(subdir) and not os.path.exists(subdir):
        os.mkdir(subdir)
    path = subdir / f'{name}_hkl.res'
    h = hkl_frame.copy()
    sources = [f'from_{l_}' for l_ in ascii_uppercase[:max_hkl_frame_count]]
    for i, source in enumerate(sources):
        h.table[source] = h.table['m'] % 2**(i+1) >= 2**i
    h.table['source_count'] = h.table[sources].sum(axis=1)
    h.table['m'] = h.table['source_count'] * max_hkl_frame_count
    for i, source in enumerate(sources):
        only_from_source = h.table[source] & (h.table['source_count'] == 1)
        h.table.loc[only_from_source, 'm'] += i
    h.to_res(path=path)


def main():
    sg = SG[space_group]
    pg = sg.reciprocate()

    h_base = HklFrame()
    h_base.edit_cell(**unit_cell)
    h_base.fill(radius=resolution)
    h_base.extinct(space_group=sg)
    h_base.find_equivalents(point_group=pg)
    hkl_stats = HklStats(h_base)

    rsd_list = []
    for rsd in reciprocal_space_direction_perp_to_dac:
        if isinstance(rsd, list):
            rsd_list.append(rsd)
        elif isinstance(rsd, tuple):
            r = h_base.a_w * rsd[0] + h_base.b_w * rsd[1] + h_base.c_w * rsd[2]
            rsd_list.append(list(r))
        else:
            raise TypeError('Use only lists [] or tuples ()')

    raw_hkl_frames = []
    for i, vector in enumerate(rsd_list):
        h_raw = h_base.copy()
        h_raw.index = i
        h_raw.name = ascii_uppercase[i]
        h_raw.dac_trim(opening_angle, vector)
        h_raw.table['F'] = 1000 + i
        h_raw.table['m'] = 2**i
        raw_hkl_frames.append(h_raw)

    for raw_hkl_frames_subset in powerset(raw_hkl_frames):
        name = ''.join(h.name for h in raw_hkl_frames_subset)

        unfolded_hkl_frames_subset = [h.copy() for h in raw_hkl_frames_subset]
        for h in unfolded_hkl_frames_subset:
            m = min(h.table['m'])
            h.transform([op.tf for op in pg.operations])
            h.merge()
            h.merge()
            h.table['m'] = m

        hkl_sum = add_hkl_frames(raw_hkl_frames_subset)
        hkl_stats.register(name, hkl_sum)
        hkl_sum.merge()
        save_nice_hkl_res(hkl_sum, name)

        hkl_unfolded_sum = add_hkl_frames(unfolded_hkl_frames_subset)
        hkl_unfolded_sum.merge()
        save_nice_hkl_res(hkl_unfolded_sum, name + '_unfolded')

        hkl_sum.find_equivalents(pg)
        equivs = set(hkl_sum.table['equiv'])
        h_missing = h_base.copy()
        missing = np.array([e not in equivs for e in h_base.table['equiv']])
        h_missing.table = h_missing.table[missing]
        h_missing.table['m'] = 1
        h_missing.table['F'] = np.linspace(1.0, 2.0, len(h_missing))
        save_nice_hkl_res(h_missing, name + '_missing')

    print(hkl_stats.table)


if __name__ == '__main__':
    main()
