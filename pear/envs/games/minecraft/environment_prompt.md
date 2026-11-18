You see the first-person Minecraft RGB frame. Aim with the crosshair at the moving red sheep. The only inputs are relative mouse motion and a left click; there is no walking, jumping, or direct entity selection.

Use `{"mouse":{"dx":0,"dy":0,"left":0}}`, replacing `dx` and `dy` with mouse counts (not pixels or angles). Positive `dx` turns right, positive `dy` looks down. Set `left` to `1` to click once or `0` for no click. One action advances exactly one game tick; the world does not evolve while you decide. The server confirms hits. The episode ends on a hit or when the tick budget runs out.
