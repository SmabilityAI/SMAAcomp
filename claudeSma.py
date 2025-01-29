import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
import scipy.stats as stats

def load_and_preprocess_data(cca_file, ruoa_file, ibero_file):
    """
    Load and preprocess data from multiple sources
    """
    # CCA Station data
    df_cca = pd.read_csv(cca_file)
    df_cca['Fecha'] = pd.to_datetime(df_cca['Fecha']) + pd.to_timedelta(df_cca.Hora, unit='h')
    df_cca['ccao3'] = df_cca['ccao3'].interpolate()
    
    # RUOA Station data
    df_ruoa = pd.read_csv(ruoa_file)
    df_ruoa['TIMESTAMP'] = pd.to_datetime(df_ruoa['TIMESTAMP'], dayfirst=True)
    df_ruoa_ = df_ruoa.set_index('TIMESTAMP').resample('60T').mean()
    df_ruoa_['RH_Avg'] = df_ruoa_['RH_Avg'].interpolate()
    df_ruoa_['Temp_Avg'] = df_ruoa_['Temp_Avg'].interpolate()
    
    # IBERO2 data
    df_ibero = pd.read_csv(ibero_file) #ibero2, hours = 1, ibero4, hours = 0
    df_ibero['Timestamp'] = pd.to_datetime(df_ibero['Timestamp'], dayfirst=True) + pd.Timedelta(hours=1) 
    df_ibero_ = df_ibero.set_index('Timestamp').resample('60T').mean()
    df_ibero_ = df_ibero_.loc['2023-03-07 15:01:00':'2023-04-04 14:47:00']
    df_ibero_['Ozone_Data'] = df_ibero_['Ozone_Data'].interpolate()
    df_ibero_['Temperature_Data'] = df_ibero_['Temperature_Data'].interpolate()
    df_ibero_['Relative_Humidity_Data'] = df_ibero_['Relative_Humidity_Data'].interpolate()
    
    return df_cca, df_ruoa_, df_ibero_

def plot_data_comparisons(df_cca, df_ruoa, df_ibero):
    """
    Create comprehensive plots for comparing data from different sources
    """
    plt.figure(figsize=(15, 12))
    
    # Plotting configurations
    plots = [
        (df_ruoa.index, df_ruoa['Temp_Avg'], df_ibero.index, df_ibero['Temperature_Data'], 
         'Temperature Comparison', 'Temperature', 'RUOA', 'Other'),
        (df_ruoa.index, df_ruoa['RH_Avg'], df_ibero.index, df_ibero['Relative_Humidity_Data'], 
         'Relative Humidity Comparison', 'Relative Humidity', 'RUOA', 'Other'),
        (df_cca['Fecha'], df_cca['ccao3'], df_ibero.index, df_ibero['Ozone_Data'], 
         'Ozone Concentration Comparison', 'Ozone Concentration', 'CCA', 'Other')
    ]
    
    for i, (x1, y1, x2, y2, title, ylabel, label1, label2) in enumerate(plots, 1):
        plt.subplot(3, 1, i)
        plt.plot(x1, y1, label=label1)
        plt.plot(x2, y2, label=label2)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel(ylabel)
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def compensate_ozone_data(ibero_ozone, temp, humidity):
    """
    Apply temperature and humidity compensation to Ozone data
    """
    compensated_ozone = []
    for o3, t, h in zip(ibero_ozone, temp, humidity):
        compensated = (o3/1.451) + (
            0.1034*t**2 - 2.225*t + 
            0.01223*h**2 - 1.984*h + 79
        )
        compensated_ozone.append(compensated)
    
    return compensated_ozone

def two_point_calibration(raw_data, reference_data):
    """
    Perform two-point calibration
    """
    raw_low, raw_high = min(raw_data), max(raw_data)
    ref_low, ref_high = min(reference_data), max(reference_data)
    
    print ("Perform two-point calibration. Raw Low and Raw High are values after compensation")
    print ("RawLow:",raw_low)
    print ("RawHigh:",raw_high)
    RawRange = (raw_high - raw_low)
    print ("RawRange:",RawRange)

    print ("ReferenceLow:",ref_low)
    print ("ReferenceHigh:",ref_high)
    ReferenceRange = (ref_high - ref_low)
    print ("ReferenceRange:",ReferenceRange)
    
    return [((x - raw_low) * (ref_high - ref_low) / (raw_high - raw_low)) + ref_low 
            for x in raw_data]

def calculate_error_metrics(reference, predicted):
    """
    Calculate various error metrics
    """
    n = len(reference)
    abs_errors = np.abs(np.array(reference) - np.array(predicted))
    
    mae = np.mean(abs_errors)
    mape = np.mean(abs_errors / np.array(reference)) * 100
    rmse = np.sqrt(np.mean(abs_errors**2))
    
    return mae, mape, rmse

def plot_final_ozone_comparison(df_cca, cca_o3, compensated_o3, calibrated_o3):
    """
    Plot the final comparison of Ozone data
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(df_cca['Fecha'], cca_o3, label='Original CCA O3', color='blue')
    plt.plot(df_cca['Fecha'], compensated_o3, label='Temperature & RH Compensated O3', color='orange')
    plt.plot(df_cca['Fecha'], calibrated_o3, label='Calibrated O3', color='green')
    
    plt.title('Ozone Concentration Comparison')
    plt.xlabel('Time')
    plt.ylabel('Ozone Concentration')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_correlation(reference, measured, title, filename=None):
    """
    Create a correlation plot with linear regression line and equation
    
    Parameters:
    reference (array-like): Reference data (e.g., CCA Ozone)
    measured (array-like): Measured/compensated data
    title (str): Title of the plot
    filename (str, optional): File to save the plot
    """
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(reference, measured)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(reference, measured, alpha=0.5)
    
    # Perfect correlation line
    min_val = min(min(reference), min(measured))
    max_val = max(max(reference), max(measured))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Correlation')
    
    # Regression line
    line = slope * np.array(reference) + intercept
    plt.plot(reference, line, 'g-', label='Linear Regression')
    
    # Formatting
    plt.xlabel('CCA Ozone Reference')
    plt.ylabel('Ibero Ozone')
    plt.title(title)
    
    # Annotation box with regression details
    equation_text = (
        f'y = {slope:.4f}x + {intercept:.4f}\n'
        f'R² = {r_value**2:.4f}\n'
        f'p-value = {p_value:.4e}'
    )
    
    # Position the text box
    plt.text(0.05, 0.95, equation_text, 
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save or show the plot
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()
    
def plot_time_series_comparison(df_cca, cca_o3, ibero_o3, compensated_o3, calibrated_o3):
    """
    Create a two-panel time series plot comparing reference and processed data
    
    Parameters:
    df_cca (DataFrame): CCA DataFrame with timestamp
    cca_o3 (array): CCA Ozone reference data
    ibero_o3 (array): Original Ibero Ozone data
    compensated_o3 (array): Temperature and humidity compensated data
    calibrated_o3 (array): Calibrated data after compensation
    """
    plt.figure(figsize=(12, 10))
    
    # Top subplot - Original vs Reference
    plt.subplot(2, 1, 1)
    plt.plot(df_cca['Fecha'], cca_o3, label='CCA Ozone Reference', color='blue')
    plt.plot(df_cca['Fecha'], ibero_o3, label='Original Ibero Ozone', color='red')
    plt.title('Original Ibero Ozone vs CCA Reference')
    plt.xlabel('Time')
    plt.ylabel('Ozone Concentration')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Bottom subplot - Compensated and Calibrated vs Reference
    plt.subplot(2, 1, 2)
    plt.plot(df_cca['Fecha'], cca_o3, label='CCA Ozone Reference', color='blue')
    plt.plot(df_cca['Fecha'], compensated_o3, label='Temperature & RH Compensated', color='orange')
    plt.plot(df_cca['Fecha'], calibrated_o3, label='Compensated & Calibrated', color='green')
    plt.title('Processed Ibero Ozone vs CCA Reference')
    plt.xlabel('Time')
    plt.ylabel('Ozone Concentration')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
def error_analysis_and_visualization(cca_O3, Ibero2_O3, tmpsma_avg, rhsma_avg):
    """
    Perform comprehensive error analysis and visualization for O3 measurements
    
    Parameters:
    - cca_O3: Reference O3 measurements
    - Ibero2_O3: O3 measurements to analyze
    - tmpsma_avg: Temperature moving average
    - rhsma_avg: Relative humidity moving average
    
    Returns:
    - Dictionary with analysis results
    """
    # Convert inputs to numpy arrays
    cca_O3 = np.array(cca_O3)
    Ibero2_O3 = np.array(Ibero2_O3)
    tmpsma_avg = np.array(tmpsma_avg)
    rhsma_avg = np.array(rhsma_avg)
    
    # Calculate absolute error (ppb)
    error_ppb = cca_O3 - Ibero2_O3
    
    # Calculate percentage error
    error_percentage = 100 * (cca_O3 - Ibero2_O3) / cca_O3
    
    # Temperature Error Analysis Figures
    plt.figure(figsize=(16, 10))
    
    # Absolute Error (ppb) vs Temperature
    plt.subplot(2, 2, 1)
    plt.scatter(tmpsma_avg, error_ppb, alpha=0.5)
    m_temp_ppb, b_temp_ppb = np.polyfit(tmpsma_avg, error_ppb, deg=1)
    plt.axline(xy1=(0, b_temp_ppb), slope=m_temp_ppb, color='r')
    
    # Linear equation text box for ppb error
    ppb_eq = f'y = {m_temp_ppb:.4f}x {b_temp_ppb:+.4f}'
    plt.text(0.05, 0.95, ppb_eq, 
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.title('Absolute Error (ppb) vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Error (ppb)')
    
    # Percentage Error vs Temperature
    plt.subplot(2, 2, 2)
    plt.scatter(tmpsma_avg, error_percentage, alpha=0.5)
    m_temp_pct, b_temp_pct = np.polyfit(tmpsma_avg, error_percentage, deg=1)
    plt.axline(xy1=(0, b_temp_pct), slope=m_temp_pct, color='r')
    
    # Linear equation text box for percentage error
    pct_eq = f'y = {m_temp_pct:.4f}x {b_temp_pct:+.4f}'
    plt.text(0.05, 0.95, pct_eq, 
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.title('Percentage Error vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Error (%)')
    
    # Quadratic fits for Temperature
    # Absolute Error
    plt.subplot(2, 2, 3)
    temp_ppb_coeffs = np.polyfit(tmpsma_avg, error_ppb, 2)
    temp_ppb_poly = np.poly1d(temp_ppb_coeffs)
    plt.scatter(tmpsma_avg, error_ppb)
    plt.plot(tmpsma_avg, temp_ppb_poly(tmpsma_avg), color='red')
    
    quad_ppb_eq = f'y = {temp_ppb_coeffs[0]:.4f}x² '\
                  f'{temp_ppb_coeffs[1]:+.4f}x '\
                  f'{temp_ppb_coeffs[2]:+.4f}'
    plt.text(0.05, 0.95, quad_ppb_eq, 
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.title('Quadratic Absolute Error (ppb) vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Error (ppb)')
    
    # Percentage Error
    plt.subplot(2, 2, 4)
    temp_pct_coeffs = np.polyfit(tmpsma_avg, error_percentage, 2)
    temp_pct_poly = np.poly1d(temp_pct_coeffs)
    plt.scatter(tmpsma_avg, error_percentage)
    plt.plot(tmpsma_avg, temp_pct_poly(tmpsma_avg), color='red')
    
    quad_pct_eq = f'y = {temp_pct_coeffs[0]:.4f}x² '\
                  f'{temp_pct_coeffs[1]:+.4f}x '\
                  f'{temp_pct_coeffs[2]:+.4f}'
    plt.text(0.05, 0.95, quad_pct_eq, 
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.title('Quadratic Percentage Error vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Error (%)')
    
    plt.tight_layout()
    plt.show()
    
    # Relative Humidity Error Analysis Figures
    plt.figure(figsize=(16, 10))
    
    # Absolute Error (ppb) vs Relative Humidity
    plt.subplot(2, 2, 1)
    plt.scatter(rhsma_avg, error_ppb, alpha=0.5)
    m_rh_ppb, b_rh_ppb = np.polyfit(rhsma_avg, error_ppb, deg=1)
    plt.axline(xy1=(0, b_rh_ppb), slope=m_rh_ppb, color='r')
    
    ppb_rh_eq = f'y = {m_rh_ppb:.4f}x {b_rh_ppb:+.4f}'
    plt.text(0.05, 0.95, ppb_rh_eq, 
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.title('Absolute Error (ppb) vs Relative Humidity')
    plt.xlabel('Relative Humidity')
    plt.ylabel('Error (ppb)')
    
    # Percentage Error vs Relative Humidity
    plt.subplot(2, 2, 2)
    plt.scatter(rhsma_avg, error_percentage, alpha=0.5)
    m_rh_pct, b_rh_pct = np.polyfit(rhsma_avg, error_percentage, deg=1)
    plt.axline(xy1=(0, b_rh_pct), slope=m_rh_pct, color='r')
    
    pct_rh_eq = f'y = {m_rh_pct:.4f}x {b_rh_pct:+.4f}'
    plt.text(0.05, 0.95, pct_rh_eq, 
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.title('Percentage Error vs Relative Humidity')
    plt.xlabel('Relative Humidity')
    plt.ylabel('Error (%)')
    
    # Quadratic fits for Relative Humidity
    # Absolute Error
    plt.subplot(2, 2, 3)
    rh_ppb_coeffs = np.polyfit(rhsma_avg, error_ppb, 2)
    rh_ppb_poly = np.poly1d(rh_ppb_coeffs)
    plt.scatter(rhsma_avg, error_ppb)
    plt.plot(rhsma_avg, rh_ppb_poly(rhsma_avg), color='red')
    
    quad_rh_ppb_eq = f'y = {rh_ppb_coeffs[0]:.4f}x² '\
                     f'{rh_ppb_coeffs[1]:+.4f}x '\
                     f'{rh_ppb_coeffs[2]:+.4f}'
    plt.text(0.05, 0.95, quad_rh_ppb_eq, 
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.title('Quadratic Absolute Error (ppb) vs Relative Humidity')
    plt.xlabel('Relative Humidity')
    plt.ylabel('Error (ppb)')
    
    # Percentage Error
    plt.subplot(2, 2, 4)
    rh_pct_coeffs = np.polyfit(rhsma_avg, error_percentage, 2)
    rh_pct_poly = np.poly1d(rh_pct_coeffs)
    plt.scatter(rhsma_avg, error_percentage)
    plt.plot(rhsma_avg, rh_pct_poly(rhsma_avg), color='red')
    
    quad_rh_pct_eq = f'y = {rh_pct_coeffs[0]:.4f}x² '\
                     f'{rh_pct_coeffs[1]:+.4f}x '\
                     f'{rh_pct_coeffs[2]:+.4f}'
    plt.text(0.05, 0.95, quad_rh_pct_eq, 
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.title('Quadratic Percentage Error vs Relative Humidity')
    plt.xlabel('Relative Humidity')
    plt.ylabel('Error (%)')
    
    plt.tight_layout()
    plt.show()
    
    # Print some error statistics
    print("Error Statistics:")
    print(f"Mean Absolute Error (ppb): {np.mean(np.abs(error_ppb)):.4f}")
    print(f"Mean Percentage Error (%): {np.mean(np.abs(error_percentage)):.4f}")
    
    return {
        'temp_linear_ppb_coef': (m_temp_ppb, b_temp_ppb),
        'temp_linear_pct_coef': (m_temp_pct, b_temp_pct),
        'temp_quad_ppb_coef': temp_ppb_coeffs,
        'temp_quad_pct_coef': temp_pct_coeffs,
        'rh_linear_ppb_coef': (m_rh_ppb, b_rh_ppb),
        'rh_linear_pct_coef': (m_rh_pct, b_rh_pct),
        'rh_quad_ppb_coef': rh_ppb_coeffs,
        'rh_quad_pct_coef': rh_pct_coeffs
    }

def main():
    # Load and preprocess data
    df_cca, df_ruoa, df_ibero = load_and_preprocess_data(
        'cca_o3_co_2023-04-11.csv', 
        '2023-03-unam_hora_L1.csv', 
        'ibero2_0703_0403.csv'       #ibero2_0703_0403.csv  ibero2_0703_0403.csv
    )
    
    # Extract relevant data
    cca_o3 = df_cca['ccao3'].values
    ibero_o3 = df_ibero['Ozone_Data'].values
    ibero_temp = df_ibero['Temperature_Data'].values
    ibero_humidity = df_ibero['Relative_Humidity_Data'].values
    
    # Correlation analysis
    print("Initial Correlation:")
    print(np.corrcoef(cca_o3, ibero_o3))
    
    # Temperature and humidity compensation
    compensated_o3 = compensate_ozone_data(ibero_o3, ibero_temp, ibero_humidity)
    
    # After compensation and before calibration, add this line
    error_analysis_results = error_analysis_and_visualization(
        cca_O3=cca_o3,  # Reference measurements
        Ibero2_O3=ibero_o3,  # Measurements to calibrate
        tmpsma_avg=ibero_temp,  # Temperature data
        rhsma_avg=ibero_humidity  # Humidity data
    )
    
    
    # Two-point calibration
    calibrated_o3 = two_point_calibration(compensated_o3, cca_o3)
    
    # Error metrics before and after compensation
    print("\nBefore Compensation:")
    mae, mape, rmse = calculate_error_metrics(cca_o3, ibero_o3)
    print(f"MAE: {mae:.4f}, MAPE: {mape:.4f}%, RMSE: {rmse:.4f}")
    
    print("\nAfter Compensation:")
    mae, mape, rmse = calculate_error_metrics(cca_o3, compensated_o3)
    print(f"MAE: {mae:.4f}, MAPE: {mape:.4f}%, RMSE: {rmse:.4f}")
    
    print("\nAfter Calibration & Compensation:")
    mae, mape, rmse = calculate_error_metrics(cca_o3, calibrated_o3)
    print(f"MAE: {mae:.4f}, MAPE: {mape:.4f}%, RMSE: {rmse:.4f}")
    
    # Pre-final correlation
    print("\nPre-final Correlation Compensation:")
    print(stats.pearsonr(cca_o3, compensated_o3))
    
    # Final correlation
    print("\nFinal Correlation Calibration & Compensation:")
    print(stats.pearsonr(cca_o3, calibrated_o3))
    
    # Optional: Plotting
    plot_data_comparisons(df_cca, df_ruoa, df_ibero)
    
    # Final Ozone Comparison Plot
    plot_final_ozone_comparison(df_cca, cca_o3, compensated_o3, calibrated_o3)
    
    
    # Add this line after other plotting functions
    plot_time_series_comparison(df_cca, cca_o3, ibero_o3, compensated_o3, calibrated_o3)
    
    # Before compensation correlation plot
    plot_correlation(
            cca_o3, 
            ibero_o3, 
            'Correlation Before Compensation', 
            'correlation_before_compensation.png'
    )
    
    # After compensation correlation plot
    plot_correlation(
        cca_o3, 
        compensated_o3, 
        'Correlation After Compensation', 
        'correlation_after_compensation.png'
    )
    
    # After compensation and calibration correlation plot
    plot_correlation(
        cca_o3, 
        calibrated_o3, 
        'Correlation After Compensation & Calibration', 
        'correlation_after_compensation.png'
    )
    
    

if __name__ == "__main__":
    main()