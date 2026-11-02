"""
ADR v3, chosen with adr_version=3. The v2 files stay as they are; v3 subclasses them.

    monitor.py       step judge: task progress kept apart from the agent's intention, zoomed crops, a step that
                     brought the task closer is not a failure; done-check with the start screen and
                     met / not met / cannot tell; click check told the action has not happened yet
    diagnostician.py two rules: type II only if nothing changed; N when the step did not really fail
    strategist.py    wait and look again before repeating; never repeat a toggle; no second blind repeat
    wrapper.py       the loop with a second look before a "no effect" flag counts, and the start screen for the
                     done-check
    replay.py        the v3 checks run offline over the recorded steps of a finished run
"""
