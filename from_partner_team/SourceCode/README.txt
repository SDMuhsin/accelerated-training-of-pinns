README – Pressure, Velocity, and Temperature Simulation

This program simulates pressure, velocity, and temperature propagation
inside a 2D pipe/domain defined by an input image.

--------------------------------------------------
1. REQUIRED INPUTS (USER MUST DEFINE THESE)
--------------------------------------------------

Before running the code, the user MUST define the following inputs
in the main function call:

1) Image file name
- The image represents the solution domain.
- Dark pixels = inside the domain
- White (or very bright) pixels = outside the domain

Example:
picture_name = "Pipe6.png"

2) Physical width of the image (in millimeters)
- This is the real-world width corresponding to the image width (x-direction).

Example:
width_mm = 2000.0

3) Physical height of the image (in millimeters)
- This is the real-world height corresponding to the image height (y-direction).

Example:
height_mm = 1000.0

4) Inlet position (in millimeters)
- Given as (x_mm, y_mm)
- Coordinates are in physical space, NOT pixel indices.
- Origin (0, 0) is at the top-left corner of the image.
- x increases to the right, y increases downward.

Example:
inlet_xy_mm = (10.0, 50.0)

5) Outlet position (in millimeters)
- Given as (x_mm, y_mm)
- Same coordinate system as the inlet.

Example:
outlet_xy_mm = (1900.0, 900.0)

IMPORTANT:
If the inlet or outlet point lies outside the solution domain,
the code will automatically snap it to the nearest point
inside the domain.

--------------------------------------------------
2. OUTPUT FILES
--------------------------------------------------

The code generates THREE CSV files:

1) Pressure CSV
File name (default):
pressure_full_simulation.csv

Columns:
x_mm, y_mm, time_s, Pressure_bar

2) Velocity CSV
File name (default):
velocity_full_simulation.csv

Columns:
x_mm, y_mm, time_s, v

Note:
Velocity is an approximate quantity computed from
the spatial gradient of pressure.

3) Temperature CSV
File name (default):
temperature_results.csv

Columns:
x_mm, y_mm, t_s, T

--------------------------------------------------
3. HOW THE DOMAIN IS INTERPRETED
--------------------------------------------------

- The input image is converted to grayscale.
- Pixels darker than a threshold are treated as INSIDE the domain.
- Bright/white pixels are treated as OUTSIDE the domain.
- The simulation only occurs inside the domain.

--------------------------------------------------
4. HOW INLET AND OUTLET ARE USED
--------------------------------------------------

- The inlet point is where pressure propagation starts.
- The outlet point is where pressure is pinned to the outlet value.
- Temperature spreading also starts from the inlet location.

--------------------------------------------------
5. HOW TO RUN
--------------------------------------------------

1) Place the image file in the same folder as the Python script.
2) Open the script and edit the following parameters in the
main function call:

picture_name
width_mm
height_mm
inlet_xy_mm
outlet_xy_mm

3) Run the script using Python.

--------------------------------------------------
6. NOTES
--------------------------------------------------

- Units:
Length : millimeters (mm)
Time : seconds (s)
Pressure: bar
Velocity: bar/mm
Temperature: arbitrary units (e.g., °C)

- Large images and long simulations may produce large CSV files.

--------------------------------------------------
END OF README
--------------------------------------------------
