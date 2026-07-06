"""Environment Explorer App — scans the /testbed workspace to provide
situational awareness before task execution.
"""

INTRO = "env_explorer: scans the /testbed workspace (data, emails, calendar) to report what files and resources exist."

from . import scan_testbed
from . import read_snippet
