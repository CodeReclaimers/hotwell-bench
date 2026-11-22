"""Step detectability thresholds under the P02 rules (GENERATED; do not edit by hand).

Written by scripts/window_sweep.py; that script's docstring defines the setup,
the grid and the threshold definition. Values are deviations from healthy in
the units of presets.THRESHOLDS_V040 (mechanisms), engineering units (sensor
bias) and noise-sigma factors (noise_increase).
"""
from __future__ import annotations

from types import MappingProxyType

RESULTS_SHA256 = '7f2d038b6ce40152877caf6a2ac371fdf8969f97ebbde6bf09e882898fb67763'
PRESET = 'P02'
WINDOW_STATS_SHA256 = 'acce80ebe4556f1ec7c1dc8e29bcac5dc6614446cad59d473dd58d8c1c2f4a1d'

THRESHOLDS_V020 = MappingProxyType({
    ('pump_degrade', 'fp1'): (
        0.003535533905932738,
        'swept, P02 rules (0.0025 not detected)',
    ),
    ('pump_degrade', 'fp2'): (
        0.003535533905932738,
        'assumed: fp1',
    ),
    ('pump_degrade', 'cp1'): (
        0.007071067811865476,
        'swept, P02 rules (0.005 not detected)',
    ),
    ('pump_degrade', 'cp2'): (
        0.007071067811865476,
        'assumed: cp1',
    ),
    ('valve_degrade', 'frv'): (
        0.014142135623730952,
        'swept, P02 rules (0.01 not detected)',
    ),
    ('valve_degrade', 'lcv'): (
        0.014142135623730952,
        'swept, P02 rules (0.01 not detected)',
    ),
    ('valve_degrade', 'makeup_valve'): (
        0.014142135623730952,
        'assumed: frv/lcv; valve normally shut',
    ),
    ('firing_degrade', ''): (
        0.01,
        'swept, P02 rules (0.00707107 not detected)',
    ),
    ('condenser_fouling', ''): (
        0.04419417382415922,
        'swept, P02 rules (0.03125 not detected)',
    ),
    ('tube_leak', ''): (
        0.01767766952966369,
        'swept, P02 rules (0.0125 not detected)',
    ),
})

SENSOR_BIAS_THRESHOLD = MappingProxyType({
    'drum_pressure': (
        31819.805153394642,
        'both signs: + swept, P02 rules (22500 not detected); - swept, P02 rules (22500 n'
        'ot detected)',
    ),
    'drum_level': (
        0.0031819805153394643,
        'both signs: + swept, P02 rules (0.00225 not detected); - swept, P02 rules (0.002'
        '25 not detected)',
    ),
    'steam_flow': (
        0.22499999999999998,
        'both signs: + swept, P02 rules (0.159099 not detected); - swept, P02 rules (0.15'
        '9099 not detected)',
    ),
    'feed_flow': (
        0.22499999999999998,
        'both signs: + swept, P02 rules (0.159099 not detected); - swept, P02 rules (0.15'
        '9099 not detected)',
    ),
    'condensate_flow': (
        0.8999999999999999,
        'both signs: + swept, P02 rules (0.636396 not detected); - swept, P02 rules (0.63'
        '6396 not detected)',
    ),
    'chest_pressure': (
        45000.0,
        'both signs: + swept, P02 rules (31819.8 not detected); - swept, P02 rules (31819'
        '.8 not detected)',
    ),
    'cond_pressure': (
        53.033008588991066,
        'both signs: + swept, P02 rules (37.5 not detected); - swept, P02 rules (37.5 not'
        ' detected)',
    ),
    'hotwell_level': (
        0.008485281374238571,
        'both signs: + swept, P02 rules (0.006 not detected); - swept, P02 rules (0.006 n'
        'ot detected)',
    ),
    'da_level': (
        0.008485281374238571,
        'both signs: + swept, P02 rules (0.006 not detected); - swept, P02 rules (0.006 n'
        'ot detected)',
    ),
    'da_pressure': (
        1414.213562373095,
        'both signs: + swept, P02 rules (1000 not detected); - swept, P02 rules (1000 not'
        ' detected)',
    ),
    'frequency': (
        None,
        'both signs: + swept, P02 rules (0.04 not detected); - swept, P02 rules (0.04 not'
        ' detected)',
    ),
    'mw_out': (
        282842.71247461904,
        'both signs: + swept, P02 rules (200000 not detected); - swept, P02 rules (200000'
        ' not detected)',
    ),
    'cp1_current': (
        0.75,
        'both signs: + swept, P02 rules (0.53033 not detected); - swept, P02 rules (0.530'
        '33 not detected)',
    ),
    'cp2_current': (
        0.1875,
        'both signs: + swept, P02 rules (0.132583 not detected); - swept, P02 rules (0.13'
        '2583 not detected)',
    ),
    'fp1_current': (
        4.242640687119286,
        'both signs: + swept, P02 rules (3 not detected); - swept, P02 rules (3 not detec'
        'ted)',
    ),
    'fp2_current': (
        0.375,
        'both signs: + swept, P02 rules (0.265165 not detected); - swept, P02 rules (0.26'
        '5165 not detected)',
    ),
    'fw_valve_pos': (
        0.01,
        'both signs: + swept, P02 rules (0.005 not detected); - swept, P02 rules (0.00707'
        '107 not detected)',
    ),
    'throttle_pos': (
        0.007071067811865476,
        'both signs: + swept, P02 rules (0.005 not detected); - swept, P02 rules (0.005 n'
        'ot detected)',
    ),
    'firing_rate': (
        0.01,
        'both signs: + swept, P02 rules (0.00707107 not detected); - swept, P02 rules (0.'
        '00707107 not detected)',
    ),
    'da_lcv_pos': (
        0.01,
        'both signs: + swept, P02 rules (0.00707107 not detected); - swept, P02 rules (0.'
        '00707107 not detected)',
    ),
    'fp1_npsh_margin': (
        0.05,
        'both signs: + swept, P02 rules (0.0353553 not detected); - swept, P02 rules (0.0'
        '353553 not detected)',
    ),
    'fp2_npsh_margin': (
        0.05,
        'both signs: + swept, P02 rules (0.0353553 not detected); - swept, P02 rules (0.0'
        '353553 not detected)',
    ),
    'makeup_flow': (
        0.01767766952966369,
        'both signs: + swept, P02 rules (0.0125 not detected); - swept, P02 rules (0.0125'
        ' not detected)',
    ),
    'makeup_valve_pos': (
        0.000625,
        'both signs: + swept, P02 rules (0.000441942 not detected); - swept, P02 rules (0'
        '.000441942 not detected)',
    ),
    'cp1_npsh_margin': (
        0.05,
        'both signs: + swept, P02 rules (0.0353553 not detected); - swept, P02 rules (0.0'
        '353553 not detected)',
    ),
    'cp2_npsh_margin': (
        0.05,
        'both signs: + swept, P02 rules (0.0353553 not detected); - swept, P02 rules (0.0'
        '353553 not detected)',
    ),
})

SENSOR_NOISE_THRESHOLD = MappingProxyType({
    'drum_pressure': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'drum_level': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'steam_flow': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'feed_flow': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'condensate_flow': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'chest_pressure': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'cond_pressure': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'hotwell_level': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'da_level': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'da_pressure': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'frequency': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'mw_out': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'cp1_current': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'cp2_current': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'fp1_current': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'fp2_current': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'fw_valve_pos': (
        2.0,
        'swept, P02 rules (1.5 not detected)',
    ),
    'throttle_pos': (
        1.5,
        'swept, P02 rules (1.3 not detected)',
    ),
    'firing_rate': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'da_lcv_pos': (
        2.0,
        'swept, P02 rules (1.5 not detected)',
    ),
    'fp1_npsh_margin': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'fp2_npsh_margin': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'makeup_flow': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'makeup_valve_pos': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'cp1_npsh_margin': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
    'cp2_npsh_margin': (
        1.3,
        'swept, P02 rules (1.2 not detected)',
    ),
})
