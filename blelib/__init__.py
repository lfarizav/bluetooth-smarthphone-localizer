"""lab-06-ble-live — BLE proximity hunt.

The package splits hard along one line: everything that can be tested without a
radio (``reading``, ``signal``, ``pathloss``, ``sim``, ``hunt``) imports nothing
from the modules that talk to hardware (``scan``, ``hci``) or to a socket
(``weblive``). That is what lets the whole lab -- including the graded
decision task -- run and be validated on a machine with no Bluetooth at all.
"""

from __future__ import annotations

__version__ = "1.0.0"
