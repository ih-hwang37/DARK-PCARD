"""Run DARK with multiple independently transformed views per input sample.

The training implementation lives in ``main.py``. This small entry point keeps
the multi-angle command discoverable without maintaining a second copy of the
training loop. An explicit ``--num_angles_per_sample`` value still overrides
the default used here.
"""

import sys

from main import main


if __name__ == "__main__":
    if "--num_angles_per_sample" not in sys.argv:
        sys.argv.extend(["--num_angles_per_sample", "5"])
    main()
