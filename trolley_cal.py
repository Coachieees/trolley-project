import itertools
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#------------------------------------------------------------------#
#                           FUNCTIONS
#------------------------------------------------------------------#
def get_placements(items):
    n = len(items)
    placements = []

    # Case 1: All items on the platform, trolley is empty
    placements.append([tuple(items), ()])
    
    # Case 2: All items on the trolley, platform is empty
    placements.append([(), tuple(items)])

    for i in range(1, n):
        for group1 in itertools.combinations(items, i):
            group1_set = set(group1)
            group2 = tuple(item for item in items if item not in group1_set)
            if len(group1) != n and len(group2) != n:
                 placements.append([group1, group2])
    
    unique_placements = list({tuple(frozenset(p) for p in place) for place in placements})
    final_placements = [[tuple(list(s)[0]), tuple(list(s)[1])] for s in unique_placements]

    return final_placements

def trolley_dynamics(t, S, m_t, m_a, m_w, theta, g):
    x, v = S
    dvdt = (m_w * g * math.cos(theta)) / (m_t + m_a + m_w * math.cos(theta)**2)
    dxdt = v
    return [dxdt, dvdt]

def euler_solver(ode_func, t_span, y0, h, args=()):
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    y_values = np.zeros((len(y0), len(t_values)))
    y_values[:, 0] = y0

    for i in range(len(t_values) - 1):
        t = t_values[i]; y = y_values[:, i]
        y_values[:, i + 1] = y + h * np.array(ode_func(t, y, *args))
    
    return t_values, y_values

def rk2_midpoint_solver(ode_func, t_span, y0, h, args=()):
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    y_values = np.zeros((len(y0), len(t_values)))
    y_values[:, 0] = y0

    for i in range(len(t_values) - 1):
        t = t_values[i]; y = y_values[:, i]
        k1 = h * np.array(ode_func(t, y, *args))
        k2 = h * np.array(ode_func(t + h / 2, y + k1 / 2, *args))
        y_values[:, i + 1] = y + k2
        
    return t_values, y_values

def rk4_solver(ode_func, t_span, y0, h, args=()):
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    y_values = np.zeros((len(y0), len(t_values)))
    y_values[:, 0] = y0

    for i in range(len(t_values) - 1):
        t = t_values[i]; y = y_values[:, i]
        k1 = h * np.array(ode_func(t, y, *args))
        k2 = h * np.array(ode_func(t + h / 2, y + k1 / 2, *args))
        k3 = h * np.array(ode_func(t + h / 2, y + k2 / 2, *args))
        k4 = h * np.array(ode_func(t + h, y + k3, *args))
        y_values[:, i + 1] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
    return t_values, y_values

def ab4_solver(ode_func, t_span, y0, h, args=()):
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    y_values = np.zeros((len(y0), len(t_values)))
    y_values[:, 0] = y0
    for i in range(3):
        t = t_values[i]; y = y_values[:, i]
        k1 = h * np.array(ode_func(t, y, *args))
        k2 = h * np.array(ode_func(t + h / 2, y + k1 / 2, *args))
        k3 = h * np.array(ode_func(t + h / 2, y + k2 / 2, *args))
        k4 = h * np.array(ode_func(t + h, y + k3, *args))
        y_values[:, i + 1] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    for i in range(3, len(t_values) - 1):
        f_i   = np.array(ode_func(t_values[i],     y_values[:, i],     *args))
        f_im1 = np.array(ode_func(t_values[i - 1], y_values[:, i - 1], *args))
        f_im2 = np.array(ode_func(t_values[i - 2], y_values[:, i - 2], *args))
        f_im3 = np.array(ode_func(t_values[i - 3], y_values[:, i - 3], *args))
        y_values[:, i + 1] = y_values[:, i] + (h / 24) * (55 * f_i - 59 * f_im1 + 37 * f_im2 - 9 * f_im3)
        
    return t_values, y_values

#------------------------------------------------------------------#
#                MECHANICAL PROPERTIES OF TROLLEY
#------------------------------------------------------------------#
m_t, x_cgt, y_cgt = 0.24602, 0.05066, 0.040222  # trolley Mass (kg) and CG (m)
L_t, H_t = 0.13500, 0.098425  # trolley Length and Height (m)
x_fr = 0.10585  # distance between front and rear wheels (m)
x_T = 0.009575  # distance between rear wheels and rope mounting point (m)
g = 9.8067
race_distance = 1

# Weight Configuration
weights_no_chosen = [1, 2, 3, 4]
weight_density, weight_dia = 7850, 10.0
weight_length = [30.0, 30.0, 40.0, 40.0, 50.0, 50.0, 60.0, 60.0, 80.0, 80.0, 100.0, 100.0, 120.0, 120.0, 200.0, 200.0]
# Calculate the mass for each weight
weight_mass = [(L / 1000) * (math.pi * ((weight_dia / 2000)**2)) * weight_density for L in weight_length]

# Holes Position
top_holes_no = [1, 2, 3, 4, 5, 6, 7]
top_first_holes_pos = 0.0225  # m
top_holes_spacing = 0.015  # m
top_holes_pos = [top_first_holes_pos + top_holes_spacing * (i - 1) for i in top_holes_no]  # Position of hole in Top Plate (m)

rear_holes_no = [1, 2, 3, 4, 5, 6, 7]
rear_first_holes_pos = 0.005425  # m
rear_holes_spacing = 0.006  # m
rear_holes_pos = [rear_first_holes_pos + rear_holes_spacing * (i - 1) for i in rear_holes_no]  # Position of hole in Rear Plate (m)

#------------------------------------------------------------------#
#                      THE BEST CASE CALCULATION
#------------------------------------------------------------------#

print(f"Chosen Weight Numbers: {weights_no_chosen}")

trolley_platform_prob = get_placements(weights_no_chosen)
print(f"{trolley_platform_prob}\n")

all_probability_calculated = []

for i in trolley_platform_prob:

    platform_caselist_ids = list(i[0])
    trolley_caselist_ids = list(i[1])
    
    platform_weightlist = [weight_mass[id - 1] for id in platform_caselist_ids]
    trolley_weightlist = [weight_mass[id - 1] for id in trolley_caselist_ids]
    
    m_w = sum(platform_weightlist)  # Total Mass on Platform
    m_a = sum(trolley_weightlist)   # Total Added Mass on Trolley

    for y_T in rear_holes_pos:  # Consider each Rope Mounting Point
        theta = math.atan((y_T - rear_first_holes_pos) / (race_distance + L_t))
        
        # Calculate Acceleration Along X-axis (acc_x)
        denominator = (m_t + m_a + m_w)
        if denominator == 0: continue
        acc_x = (m_w * g * math.cos(theta)) / denominator

        # Calculate Tension in Rope (T)
        T = m_w * (g - acc_x * math.sin(theta))

        if m_a == 0:
            N_f_numerator = (m_t * g * x_cgt) + (T * y_T * math.cos(theta)) - (T * x_T * math.sin(theta))
            N_f = N_f_numerator / x_fr
            
            prob_case = [platform_caselist_ids, [], y_T, [], acc_x, T, 0, N_f, theta, m_w, m_a]
            all_probability_calculated.append(prob_case)
            continue

        for k in itertools.permutations(top_holes_pos, len(trolley_caselist_ids)):
            x_cga_i = [trolley_weightlist[idx] * k[idx] for idx in range(len(trolley_weightlist))]
            x_cga = 0.120425 - (sum(x_cga_i) / m_a)
            
            # Calculate Force on front wheels (N_f)
            N_f_numerator = (m_t * g * x_cgt) + (m_a * g * x_cga) + (T * y_T * math.cos(theta)) - (T * x_T * math.sin(theta))
            N_f = N_f_numerator / x_fr

            prob_case = [platform_caselist_ids, trolley_caselist_ids, y_T, list(k), acc_x, T, x_cga, N_f, theta, m_w, m_a]
            all_probability_calculated.append(prob_case)

if all_probability_calculated:
    best_case = max(all_probability_calculated, key=lambda x: (x[4], x[7]))
    amount_of_prob = len(all_probability_calculated)
    print("-" * 65)
    print(f"Calculated Case: {amount_of_prob} case")
    print("-" * 65)
    print(f"Platform Weight :       {best_case[0]}      => {best_case[9]:.4f} kg")
    print(f"Trolley Weight :        {best_case[1]}      => {best_case[10]:.4f} kg")
    print(f"Weight Positions on Trolley:    {best_case[3]}")
    print(f"Rope Mounting Point:     {best_case[2]*1000:.3f} mm | theta: {math.degrees(best_case[8]):.2f} deg")

    print(f"\n---> Max Acceleration (acc_x): {best_case[4]:.4f} m/s²")
    print(f"---> Rope Tension (T): {best_case[5]:.4f} N")
    print(f"---> CG of Added Weight (x_cga): {best_case[6]:.4f} m")
    print(f"---> Force on Front Wheel (N_f): {best_case[7]:.4f} N")
    print("-" * 65)
    print()
else:
    print("No valid cases were calculated.")

#------------------------------------------------------------------#
#                       NUMERICAL SIMULATION
#------------------------------------------------------------------#

# Define simulation parameters
h = 0.05
t_span = [0, 5]
initial_conditions = [0, 0]
sim_args = (m_t, best_case[10], best_case[9], best_case[8], g)

t_eul, y_eul = euler_solver(trolley_dynamics, t_span, initial_conditions, h, args=sim_args)
t_rk2, y_rk2 = rk2_midpoint_solver(trolley_dynamics, t_span, initial_conditions, h, args=sim_args)
t_rk4, y_rk4 = rk4_solver(trolley_dynamics, t_span, initial_conditions, h, args=sim_args)
t_ab2, y_ab2 = ab4_solver(trolley_dynamics, t_span, initial_conditions, h, args=sim_args)

# Calculate finish times
time_eul = np.interp(race_distance, y_eul[0, :], t_eul)
time_rk2 = np.interp(race_distance, y_rk2[0, :], t_rk2)
time_rk4 = np.interp(race_distance, y_rk4[0, :], t_rk4)
time_ab2 = np.interp(race_distance, y_ab2[0, :], t_ab2)

# Exact
accel_exact = best_case[4]
time_exact = (2 * race_distance / accel_exact)**0.5 if accel_exact > 0 else float('inf')

print(f"Comparing finish times for a {race_distance}m race (step-size h={h}s):")
print("-" * 65)
print(f"{'Method':<20} | {'Finish Time (s)':<20} | {'Error vs Exact':<20}")
print("-" * 65)
print(f"{'Exact Solution':<20} | {time_exact:<20.8f} | {'N/A'}")
print(f"{'Euler Method':<20} | {time_eul:<20.8f} | {abs(time_eul - time_exact):.8f}")
print(f"{'RK2':<20} | {time_rk2:<20.8f} | {abs(time_rk2 - time_exact):.8f}")
print(f"{'RK4':<20} | {time_rk4:<20.8f} | {abs(time_rk4 - time_exact):.8f}")
print(f"{'Adams-Bashforth':<20} | {time_ab2:<20.8f} | {abs(time_ab2 - time_exact):.8f}")

#Plot comparison
print("\nDisplaying simulation plots...")
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 8))
plt.plot(t_eul, y_eul[0, :], '^--', label=f'Euler (h={h})', markersize=4, alpha=0.8)
plt.plot(t_rk2, y_rk2[0, :], 's--', label=f'RK2 Midpoint (h={h})', markersize=4, alpha=0.8)
plt.plot(t_rk4, y_rk4[0, :], 'd-', label=f'RK4 Classic (h={h})', markersize=5, zorder=5)
plt.plot(t_ab2, y_ab2[0, :], 'o--', label=f'Adams-Bashforth (h={h})', markersize=4, alpha=0.8)
plt.axhline(y=race_distance, color='r', linestyle='--', label=f'Finish Line ({race_distance}m)')
plt.axvline(x=time_exact, color='k', linestyle=':', label=f'Exact Time ({time_exact:.3f}s)', linewidth=2)
plt.title(f'Comparison of Numerical Methods', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Position (m)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True)
plt.xlim(0, time_exact * 1.2)
plt.ylim(0, race_distance * 1.2)
plt.show()
