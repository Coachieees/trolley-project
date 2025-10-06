import itertools
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- PREVIOUS FUNCTIONS (get_placements, accel_x_cal, etc.) ---
def get_placements(items):
    """Generates all possible ways to split a list of items into two groups."""
    n = len(items)
    if n < 2:
        return []

    placements = []
    # Loop from 1 to n-1 to ensure both groups are non-empty
    for i in range(1, n):
        for group1 in itertools.combinations(items, i):
            group1_set = set(group1)
            group2 = tuple(item for item in items if item not in group1_set)
            placements.append([group1, group2])
    return placements

def accel_x_cal(m_t, m_a, m_w, theta):
    """Calculates the acceleration in the x-direction."""
    if m_w == 0:
        return 0
    a_x = (m_w * g * math.cos(theta)) / (m_t + m_a + m_w * math.cos(theta)**2)
    return a_x

def tension_cal(m_t, m_a, acc_x, theta):
    """Calculates the tension in the string."""
    if acc_x == 0:
        return 0
    T = ((m_t + m_a) * acc_x) / (math.cos(theta))
    return T

def Nf_cal(m_t, m_a, x_cgt, x_cga, T, x_T, y_T, x_fr, theta):
    """Calculates the normal force on the front wheel."""
    x_cgt, x_cga, x_T, y_T, x_fr = x_cgt / 1000, x_cga / 1000, x_T / 1000, y_T / 1000, x_fr / 1000
    N_f = (((m_t * x_cgt + m_a * x_cga) * g) + T * (y_T * math.cos(theta) - x_T * math.sin(theta))) / x_fr
    return N_f

def run_simulation_case(case_num, m_a, m_w):
    print(f"Case{case_num} |  Added Weigth: {m_a:<9.6f} kg  |  Platform Weigth: {m_w:<9.6f} kg")

    best_case_data = {
        'max_acc_x': -1.0,
        'best_rear_hole': 0,
        'max_Nf': -float('inf'),
        'best_top_hole': 0
    }

    for rear_hole_no in range(6):
        y_T = 24.43 + (rear_hole_no * 6)
        theta = math.atan((y_T-24.43) / (race_distance * 1000 + L_t))
        print(f"   Rear Hole No.{rear_hole_no + 1} -> Theta: {math.degrees(theta):.3f} deg", end="")

        acc_x = accel_x_cal(m_t, m_a, m_w, theta)
        print(f"   ---> acc_x: {acc_x:.4f} m/s^2", end="")

        T = tension_cal(m_t, m_a, acc_x, theta)
        print(f"   -> T: {T:.4f} N")

        if acc_x > best_case_data['max_acc_x']:
            best_case_data['max_acc_x'] = acc_x
            best_case_data['best_rear_hole'] = rear_hole_no + 1
            
            current_max_Nf = -float('inf')
            current_best_top_hole = 0
            for top_hole_no in range(7):
                x_cga = 97.93 - (top_hole_no * 15)
                Nf = Nf_cal(m_t, m_a, x_cgt, x_cga, T, x_T, y_T, x_fr, theta)
                if Nf > current_max_Nf:
                    current_max_Nf = Nf
                    current_best_top_hole = top_hole_no + 1
            
            best_case_data['max_Nf'] = current_max_Nf
            best_case_data['best_top_hole'] = current_best_top_hole

        for top_hole_no in range(7):
            x_cga = 97.93 - (top_hole_no * 15)
            Nf = Nf_cal(m_t, m_a, x_cgt, x_cga, T, x_T, y_T, x_fr, theta)
            print(f"         Top Hole No.{top_hole_no + 1} -> Nf: {Nf:.4f} N")
        print("")
        
    print("------------------------------------------------------------------------------------")
    return {
        'case': case_num, 'm_a': m_a, 'm_w': m_w,
        'rear_hole': best_case_data['best_rear_hole'], 'acc_x': best_case_data['max_acc_x'],
        'top_hole': best_case_data['best_top_hole'], 'Nf': best_case_data['max_Nf']
    }

def trolley_dynamics(t, S, m_t, m_a, m_w, theta, g):
    """Defines the ODE system for the trolley."""
    x, v = S
    dvdt = (m_w * g * math.cos(theta)) / (m_t + m_a + m_w * math.cos(theta)**2)
    dxdt = v
    return [dxdt, dvdt]

#------------------------------------------------------------------#
#           IMPLEMENTATIONS OF NUMERICAL SOLVERS
#------------------------------------------------------------------#

def euler_solver(ode_func, t_span, y0, h, args=()):
    """Solves an ODE system using the Forward Euler method."""
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    y_values = np.zeros((len(y0), len(t_values)))
    y_values[:, 0] = y0

    for i in range(len(t_values) - 1):
        t = t_values[i]; y = y_values[:, i]
        y_values[:, i + 1] = y + h * np.array(ode_func(t, y, *args))
    
    return t_values, y_values

def rk2_midpoint_solver(ode_func, t_span, y0, h, args=()):
    """Solves an ODE system using the RK2 (Midpoint) method."""
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
    """Solves an ODE system using the classic 4th-order Runge-Kutta method."""
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

def ab2_solver(ode_func, t_span, y0, h, args=()):
    """Solves an ODE system using the 2-step Adams-Bashforth method."""
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    y_values = np.zeros((len(y0), len(t_values)))
    y_values[:, 0] = y0
    
    # Use RK2 for the first step to "start up" the method
    t0, y0_arr = t_values[0], y_values[:, 0]
    k1_start = h * np.array(ode_func(t0, y0_arr, *args))
    k2_start = h * np.array(ode_func(t0 + h / 2, y0_arr + k1_start / 2, *args))
    y_values[:, 1] = y0_arr + k2_start

    # Apply Adams-Bashforth for the remaining steps
    for i in range(1, len(t_values) - 1):
        t_i, y_i = t_values[i], y_values[:, i]
        t_im1, y_im1 = t_values[i - 1], y_values[:, i - 1]
        f_i = np.array(ode_func(t_i, y_i, *args))
        f_im1 = np.array(ode_func(t_im1, y_im1, *args))
        y_values[:, i + 1] = y_i + h * (1.5 * f_i - 0.5 * f_im1)
        
    return t_values, y_values

# --- CONSTANTS AND SETUP (Unchanged) ---
m_t, x_cgt, y_cgt, L_t, H_t, x_fr, g, x_T = 0.24602, 50.66, 53.22, 135.00, 98.43, 105.86, 9.8067, 14.57
race_distance = 1
weights_no_chosen = [1, 2, 3, 4]
weight_density, weight_dia = 7850, 10.0
weight_length = [30.0, 30.0, 40.0, 40.0, 50.0, 50.0, 60.0, 60.0, 80.0, 80.0, 100.0, 100.0, 120.0, 120.0, 200.0, 200.0]
weight_mass = [(L / 1000) * (math.pi * ((weight_dia / 2000)**2)) * weight_density for L in weight_length]
weight_mass_cal = [weight_mass[i - 1] for i in weights_no_chosen]
total_mass = sum(weight_mass_cal)
print(f"Given the list of masses: {weight_mass_cal}\n")

# --- SIMULATION & SUMMARY (Unchanged) ---
best_results_summary = []
best_results_summary.append(run_simulation_case(case_num=1, m_a=total_mass, m_w=0))
all_placements = get_placements(weight_mass_cal)
for i, p in enumerate(all_placements):
    m_a_current, m_w_current = sum(p[0]), sum(p[1])
    best_results_summary.append(run_simulation_case(case_num=i + 2, m_a=m_a_current, m_w=m_w_current))
final_case_num = len(all_placements) + 2
best_results_summary.append(run_simulation_case(case_num=final_case_num, m_a=0, m_w=total_mass))

print("\n\n" + "="*103)
print("                                 OPTIMAL CONFIGURATION SUMMARY")
print("="*103)
print(f"{'Case':<5} | {'Added W (kg)':<15} | {'Platform W (kg)':<17} | {'Best Rear Hole':<14} | {'Max Accel (m/s^2)':<19} | {'Best Top Hole':<13} | {'Max Nf (N)':<12}")
print("-" * 103)
for result in best_results_summary:
    print(f"{result['case']:<5} | {result['m_a']:<15.6f} | {result['m_w']:<17.6f} | {result['rear_hole']:<14} | {result['acc_x']:<19.4f} | {result['top_hole']:<13} | {result['Nf']:<12.4f}")
print("-" * 103)


#------------------------------------------------------------------#
#       UPDATED: DYNAMIC SIMULATION & METHOD COMPARISON
#------------------------------------------------------------------#
# 1. Find and configure the best case
best_case_config = max(best_results_summary, key=lambda item: item['acc_x'])
case_num, m_a_best, m_w_best, rear_hole_best = best_case_config['case'], best_case_config['m_a'], best_case_config['m_w'], best_case_config['rear_hole']
y_T_best = 18.43 + ((rear_hole_best - 1) * 6)
theta_best = math.atan(y_T_best / (race_distance * 1000 + L_t))

print(f"\n\n{'='*103}")
print(f"       DYNAMIC SIMULATION & METHOD COMPARISON FOR BEST CASE (Case {case_num})")
print(f"{'='*103}\n")

# 2. Define simulation parameters
t_span = [0, 5]
initial_conditions = [0, 0]
h = 0.05  # Fixed step-size for custom solvers
sim_args = (m_t, m_a_best, m_w_best, theta_best, g)

# 3. Solve with all methods
t_eul, y_eul = euler_solver(trolley_dynamics, t_span, initial_conditions, h, args=sim_args)
t_rk2, y_rk2 = rk2_midpoint_solver(trolley_dynamics, t_span, initial_conditions, h, args=sim_args)
t_rk4, y_rk4 = rk4_solver(trolley_dynamics, t_span, initial_conditions, h, args=sim_args)
t_ab2, y_ab2 = ab2_solver(trolley_dynamics, t_span, initial_conditions, h, args=sim_args)

# 4. Calculate finish times
time_eul = np.interp(race_distance, y_eul[0, :], t_eul)
time_rk2 = np.interp(race_distance, y_rk2[0, :], t_rk2)
time_rk4 = np.interp(race_distance, y_rk4[0, :], t_rk4)
time_ab2 = np.interp(race_distance, y_ab2[0, :], t_ab2)

# 5. Calculate the exact analytical solution time for comparison
accel_exact = best_case_config['acc_x']
time_exact = (2 * race_distance / accel_exact)**0.5 if accel_exact > 0 else float('inf')

# 6. Print comparison table
print(f"Comparing finish times for a {race_distance}m race (step-size h={h}s):\n")
print(f"{'Method':<20} | {'Finish Time (s)':<20} | {'Error vs Exact':<20}")
print("-" * 65)
print(f"{'Exact Solution':<20} | {time_exact:<20.6f} | {'N/A'}")
print(f"{'Euler Method':<20} | {time_eul:<20.6f} | {abs(time_eul - time_exact):.6f}")
print(f"{'RK2':<20} | {time_rk2:<20.6f} | {abs(time_rk2 - time_exact):.6f}")
print(f"{'RK4':<20} | {time_rk4:<20.6f} | {abs(time_rk4 - time_exact):.6f}")
print(f"{'Adams-Bashforth':<20} | {time_ab2:<20.6f} | {abs(time_ab2 - time_exact):.6f}")

# 7. Plot comparison
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
