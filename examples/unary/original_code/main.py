from errors import errors
import unary as un


S0 = 2
K = 1.9
sig = 0.4
r = 0.05
T = 0.1
data = (S0, sig, r, T, K)

bins = 8
max_error_gate = 0.005
error_name = 'thermal'
repeats = 100
measure = False
thermal = False
steps = 51
Err = errors(data, max_error_gate, steps)
'''print('binary')
Err.compute_save_errors_binary(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
print('unary')
Err.compute_save_errors_unary(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)'''
print('paint errors')
Err.paint_errors(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
'''print('KL unary')
Err.compute_save_KL_unary(8, error_name, repeats, measure_error=measure, thermal_error=thermal)
print('KL binary')
Err.compute_save_KL_binary(8, error_name, repeats, measure_error=measure, thermal_error=thermal)'''

print('paint outcomes')
Err.paint_outcomes(bins, error_name, 0.0, repeats, measure_error=measure, thermal_error=thermal)
Err.paint_outcomes(bins, error_name, 0.001, repeats, measure_error=measure, thermal_error=thermal)
print('paint divergences')
Err.paint_divergences(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)

'''print('AE unary')
Err.compute_save_errors_unary_amplitude_estimation(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
print('AE binary')
Err.compute_save_errors_binary_amplitude_estimation(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)'''

print('paint AE')
Err.error_emplitude_estimation(bins, error_name, repeats, measure_error=measure, thermal_error=thermal)
Err.paint_amplitude_estimation_binary(bins, error_name, repeats, M=4, measure_error=measure, thermal_error=thermal)
Err.paint_amplitude_estimation_unary(bins, error_name, repeats, M=4, measure_error=measure, thermal_error=thermal)





