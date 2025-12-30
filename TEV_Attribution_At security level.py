"""
===============================================================================
ENHANCED FIXED INCOME TEV ATTRIBUTION SYSTEM
===============================================================================

Features:
- 9-factor decomposition (7 rates + OAS + FX)
- EUR base currency with proper conversion
- Spread duration from sum of KRDs
- Complete validation framework
- 4-sheet Excel output (Attribution, Validation, Covariance, Program Log)

Corrections Applied:
- EUR KRD = USD KRD / FX_rate (not multiply)
- Spread duration = Sum(KRDs) × 0.95 (if callable)
- Missing data includes 0 values
- Covariance × 12 for annualization

Author: TEV Attribution Project
Date: December 2025
Version: 1.0 Final
===============================================================================
"""

import pandas as pd
import numpy as np
import QuantLib as ql
# from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Global QuantLib today
today = ql.Date.todaysDate()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Portfolio parameters
PORTFOLIO_VALUE_EUR = 1_000_000  # Portfolio value in EUR
BASE_CURRENCY = 'EUR'
VALIDATION_TOLERANCE_BPS = 0.1  # Tolerance for validation checks

# File paths (will be set in main)
RATES_FILE = "C:\\Users\\amits\\Desktop\\All quant workshop\\Market Risk\\Tracking error\\Expost TE-Enhanced, Security wide TE attribution\\All_Constant_Maturity_rates.xlsx"
HOLDINGS_FILE = "C:\\Users\\amits\\Desktop\\All quant workshop\\Market Risk\\Tracking error\\Expost TE-Enhanced, Security wide TE attribution\\Bond holdings.xlsx"
OUTPUT_FILE = 'TEV_Attribution_Results.xlsx'

# =============================================================================
# STEP 1: DATA LOADING
# =============================================================================

def load_treasury_rates_enhanced(file_path):
    """
    Load enhanced rates file with OAS and FX columns
    Expected columns: observation_date, GS1-GS10, A (OAS), EUR/USD
    """
    print("\n[1/12] Loading rates data...")
    
    rates_df = pd.read_excel(file_path)
    rates_df['observation_date'] = pd.to_datetime(rates_df['observation_date'])
    rates_df.set_index('observation_date', inplace=True)
    
    # Check required columns
    required_cols = ['GS1', 'GS2', 'GS3', 'GS4', 'GS5', 'GS07', 'GS10', 'A', 'EUR/USD']
    missing_cols = [col for col in required_cols if col not in rates_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in rates file: {missing_cols}")
    
    # Rename for consistency
    rates_df.rename(columns={
        'GS1': '1Y', 'GS2': '2Y', 'GS3': '3Y', 'GS4': '4Y',
        'GS5': '5Y', 'GS07': '7Y', 'GS10': '10Y',
        'A': 'OAS'
    }, inplace=True)
    
    # Convert from percentage to decimal if needed
    for col in ['1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y', 'OAS']:
        if rates_df[col].max() > 1:
            rates_df[col] = rates_df[col] / 100
    
    # Check for missing/zero values in last 5 years
    five_years_ago = rates_df.index[-1] - pd.DateOffset(years=5)
    recent = rates_df[rates_df.index >= five_years_ago]
    
    oas_missing = (recent['OAS'].isnull() | (recent['OAS'] == 0)).sum()
    fx_missing = (recent['EUR/USD'].isnull() | (recent['EUR/USD'] == 0)).sum()
    
    if oas_missing > 0 or fx_missing > 0:
        raise ValueError(f"Missing/zero values in last 5 years: OAS={oas_missing}, FX={fx_missing}")
    
    print(f"      Loaded {len(rates_df)} observations")
    print(f"      Date range: {rates_df.index[0].strftime('%Y-%m-%d')} to {rates_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"      ✓ No missing values in last 5 years")
    
    return rates_df

def load_bond_holdings_enhanced(file_path):
    """
    Load bond holdings with all required columns
    """
    print("\n[2/12] Loading bond holdings...")
    
    holdings_df = pd.read_excel(file_path)
    
    # Check required columns
    required_cols = ['Security', 'Portfolio/Benchmark', 'Weight', 'Notional in holding currency',
                    'Rating', 'Holding currency', 'Issue Date', 'Call date', 
                    'Maturity', 'Coupon']
    missing_cols = [col for col in required_cols if col not in holdings_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in holdings file: {missing_cols}")
    
    # Convert date columns
    for col in ['Issue Date', 'Call date', 'Maturity']:
        holdings_df[col] = pd.to_datetime(holdings_df[col])
    
    # Convert coupon to decimal if needed
    if holdings_df['Coupon'].max() > 1:
        holdings_df['Coupon'] = holdings_df['Coupon'] / 100
    
    print(f"      Loaded {len(holdings_df)} positions")
    print(f"      Portfolio positions: {len(holdings_df[holdings_df['Portfolio/Benchmark']=='Portfolio'])}")
    print(f"      Benchmark positions: {len(holdings_df[holdings_df['Portfolio/Benchmark']=='Benchmark'])}")
    
    return holdings_df

# =============================================================================
# STEP 2: COVARIANCE MATRIX (9×9 with OAS and FX)
# =============================================================================

def calculate_enhanced_covariance_matrix(rates_df, years_back=5):
    """
    Calculate 9×9 covariance matrix including OAS and FX
    """
    print("\n[3/12] Calculating enhanced covariance matrix...")
    
    # Filter to last 5 years
    end_date = rates_df.index[-1]
    start_date = end_date - pd.DateOffset(years=years_back)
    recent_rates = rates_df[rates_df.index >= start_date]
    
    print(f"      Using data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"      Observations: {len(recent_rates)}")
    
    # Resample to monthly
    monthly_data = recent_rates.resample('M').last()
    
    # Calculate monthly changes
    # Rates & OAS: absolute changes
    rate_cols = ['1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y', 'OAS']
    rate_changes = monthly_data[rate_cols].diff().dropna()
    
    # FX: percentage returns (CRITICAL: not absolute changes!)
    fx_returns = monthly_data['EUR/USD'].pct_change().dropna()
    
    # Combine into single DataFrame
    combined_changes = rate_changes.copy()
    combined_changes['FX'] = fx_returns
    
    # Calculate monthly covariance
    monthly_cov = combined_changes.cov()
    
    # Annualize (multiply by 12)
    annual_cov = monthly_cov * 12
    
    # Calculate annual volatilities
    annual_vol = combined_changes.std() * np.sqrt(12)
    
    # Calculate correlations
    correlation = combined_changes.corr()
    
    print(f"      Monthly observations: {len(combined_changes)}")
    print(f"      ✓ Covariance matrix: 9×9")
    print(f"      ✓ Annualized (×12)")
    
    return annual_cov, annual_vol, correlation, combined_changes

# =============================================================================
# STEP 3: QUANTLIB SETUP (from base code)
# =============================================================================

def setup_quantlib():
    """Initialize QuantLib settings"""
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    return calendar, today

def build_curve_from_zeros(dates, rates, calendar, settlement_days=1, face_value=100):
    """Build yield curve from zero rates"""
    bond_helpers = []
    for d, r in zip(dates, rates):
        price = 100 * np.exp(-r * ql.ActualActual(ql.ActualActual.Bond).yearFraction(today, d))
        helper_schedule = ql.Schedule(today, d, ql.Period(ql.Once), calendar, 
                                    ql.Unadjusted, ql.Unadjusted, 
                                    ql.DateGeneration.Backward, False)
        
        helper = ql.FixedRateBondHelper(
            ql.QuoteHandle(ql.SimpleQuote(price)),
            settlement_days, 
            face_value, 
            helper_schedule, 
            [0.0],  # Zero coupon rate
            ql.ActualActual(ql.ActualActual.Bond)
        )
        bond_helpers.append(helper)
    
    curve = ql.PiecewiseLinearZero(today, bond_helpers, ql.ActualActual(ql.ActualActual.Bond))
    return ql.YieldTermStructureHandle(curve)

# =============================================================================
# STEP 4: ENHANCED KRD CALCULATION
# =============================================================================

def calculate_bond_krd_enhanced(bond_data, zero_rates, calendar):
    """
    Calculate KRDs for a single bond
    Returns: dict of KRDs, base_price, is_callable
    """
    # Extract bond parameters
    issue_date = ql.Date(bond_data['Issue Date'].day, 
                        bond_data['Issue Date'].month, 
                        bond_data['Issue Date'].year)
    maturity_date = ql.Date(bond_data['Maturity'].day, 
                           bond_data['Maturity'].month, 
                           bond_data['Maturity'].year)
    coupon_rate = bond_data['Coupon']
    
    # Check if callable
    call_date = None
    is_callable = False
    if pd.notna(bond_data['Call date']):
        call_date = ql.Date(bond_data['Call date'].day, 
                           bond_data['Call date'].month, 
                           bond_data['Call date'].year)
        is_callable = True
    
    # Create bond schedule
    schedule = ql.Schedule(issue_date, maturity_date, ql.Period(ql.Annual),
                          calendar, ql.Unadjusted, ql.Unadjusted,
                          ql.DateGeneration.Backward, False)
    
    # Generate dates for all 7 tenors
    zero_dates = []
    tenor_years = [1, 2, 3, 4, 5, 7, 10]
    for years in tenor_years:
        tenor_date = today + ql.Period(years, ql.Years)
        zero_dates.append(tenor_date)
    
    # Build initial curve
    rates_list = [zero_rates[f'{y}Y'] for y in [1, 2, 3, 4, 5, 7, 10]]
    yield_curve_handle = build_curve_from_zeros(zero_dates, rates_list, calendar)
    
    # Create bond object
    if call_date:
        # Callable bond
        callability_schedule = ql.CallabilitySchedule()
        callability_schedule.append(
            ql.Callability(
                ql.BondPrice(100.0, ql.BondPrice.Clean),
                ql.Callability.Call,
                call_date
            )
        )
        bond = ql.CallableFixedRateBond(1, 100, schedule, [coupon_rate],
                                        ql.ActualActual(ql.ActualActual.Bond),
                                        ql.Following, 100.0, issue_date, 
                                        callability_schedule)
        hw_model = ql.HullWhite(yield_curve_handle, a=0.03, sigma=0.015)
        engine = ql.TreeCallableFixedRateBondEngine(hw_model, 500)
        bond.setPricingEngine(engine)
    else:
        # Regular bond
        bond = ql.FixedRateBond(1, 100, schedule, [coupon_rate],
                               ql.ActualActual(ql.ActualActual.Bond))
        engine = ql.DiscountingBondEngine(yield_curve_handle)
        bond.setPricingEngine(engine)
    
    # Calculate base price
    base_price = bond.cleanPrice()
    
    # Calculate KRDs for each tenor
    krds = {}
    shift_size = 0.0001  # 1 basis point
    
    for i, tenor_str in enumerate(['1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y']):
        # Bump rate up
        rates_up = rates_list.copy()
        rates_up[i] += shift_size
        curve_up = build_curve_from_zeros(zero_dates, rates_up, calendar)
        
        if call_date:
            hw_model_up = ql.HullWhite(curve_up, a=0.03, sigma=0.015)
            bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(hw_model_up, 500))
        else:
            bond.setPricingEngine(ql.DiscountingBondEngine(curve_up))
        price_up = bond.cleanPrice()
        
        # Bump rate down
        rates_down = rates_list.copy()
        rates_down[i] -= shift_size
        curve_down = build_curve_from_zeros(zero_dates, rates_down, calendar)
        
        if call_date:
            hw_model_down = ql.HullWhite(curve_down, a=0.03, sigma=0.015)
            bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(hw_model_down, 500))
        else:
            bond.setPricingEngine(ql.DiscountingBondEngine(curve_down))
        price_down = bond.cleanPrice()
        
        # Calculate KRD
        # Rate Sensitivity should be negative, meaning if the rate rises then the bond value goes down.
        krd = (price_up - price_down) / (2 * base_price * shift_size)
        krds[tenor_str] = krd
    
    return krds, base_price, is_callable

# =============================================================================
# STEP 5: SPREAD DURATION (Sum of KRDs)
# =============================================================================

def calculate_spread_duration(krds, is_callable):
    """
    Calculate spread duration from KRDs
    
    CORRECTION: Spread duration = Sum of all KRDs (× 0.95 if callable)
    """
    # Sum all KRDs to get modified duration
    modified_duration = sum(krds.values())
    
    # Spread duration should be negative because if OAS rises then position value goes down.
    # Adjust for callability
    if is_callable:
        spread_duration = modified_duration * 0.95
    else:
        spread_duration = modified_duration
    
    return spread_duration

# =============================================================================
# STEP 6: FX SENSITIVITY
# =============================================================================

def calculate_fx_sensitivity(bond_data, net_weight, current_fx_rate, base_price):
   """
    Calculate FX sensitivity using finite difference method
    
    CORRECTED VERSION: Uses proper finite difference calculation
    analogous to KRD calculation
    
    Parameters:
    -----------
    bond_data : Series
        Bond characteristics
    net_weight : float
        Net weight (Portfolio - Benchmark)
    current_fx_rate : float
        Current EUR/USD rate
    
    Returns:
    --------
    float : Net FX exposure (EUR change per 0.01 FX move)
    """
   holding_currency = bond_data['Holding currency']
  
    
   if holding_currency == 'USD':
        # USD bond - calculate FX sensitivity
      Bond_Price = base_price*current_fx_rate
      
      # 1% increase in FX rate
      current_fx_rate_up = current_fx_rate*1.0001
      Bond_Price_up = Bond_Price*current_fx_rate_up
      
      current_fx_rate_down = current_fx_rate*0.9999
      Bond_Price_down = Bond_Price*current_fx_rate_down
      # FX sensitivity in form of USD exposure should be positive because if USD strengthens then Position value increases in Euro
      Bond_Exchange_rate_sensitivity =  (Bond_Price_up - Bond_Price_down)/(2*Bond_Price*0.0001)
        
   else:  # EUR
        # EUR bond - no FX exposure
      Bond_Exchange_rate_sensitivity = 0.0
      
   return Bond_Exchange_rate_sensitivity

# =============================================================================
# STEP 7: EUR BASE CURRENCY CONVERSION
# =============================================================================

def convert_to_eur_base(krds, holding_currency, current_fx_rate):
    """
    Convert USD KRDs to EUR base currency
    
    CORRECTION: EUR_KRD = USD_KRD / FX_rate (DIVIDE, not multiply!)
    
    Rationale: EUR/USD = 1.10 means 1 EUR = 1.10 USD
              Higher EUR value → lower duration in EUR terms
    """
    if holding_currency == 'USD':
        # Convert USD KRDs to EUR base
        eur_krds = {}
        for tenor, krd in krds.items():
            eur_krds[tenor] = krd / current_fx_rate
    else:  # EUR
        # Already in EUR
        eur_krds = krds.copy()
    
    return eur_krds

# =============================================================================
# STEP 8: SECURITY-LEVEL TEV CALCULATION
# =============================================================================

def calculate_security_tev(bond_data, current_rates, current_oas, current_fx_rate, 
                          annual_cov, calendar, portfolio_value_eur, program_log):
    """
    Calculate complete TEV attribution for a single security
    """
    security_name = bond_data['Security']
    
    try:
        # Check if matured
        if bond_data['Maturity'] < pd.Timestamp.now():
            program_log.append({
                'Security': security_name,
                'Issue': 'Bond has matured',
                'Action': 'Skipped',
                'Details': f"Maturity: {bond_data['Maturity'].strftime('%Y-%m-%d')}"
            })
            return None
        
        # Calculate net weight (use pre-calculated values if available)
        if '_portfolio_weight' in bond_data.index:
            portfolio_weight = bond_data['_portfolio_weight']
            benchmark_weight = bond_data['_benchmark_weight']
        else:
            # Fallback: assume single entry
            if bond_data['Portfolio/Benchmark'] == 'Portfolio':
                portfolio_weight = bond_data['Weight']
                benchmark_weight = 0
            else:
                portfolio_weight = 0
                benchmark_weight = bond_data['Weight']
        
        net_weight = portfolio_weight - benchmark_weight
        
        # Calculate KRDs
        krds, base_price, is_callable = calculate_bond_krd_enhanced(bond_data, current_rates, calendar)
        
        # Convert to EUR base
        eur_krds = convert_to_eur_base(krds, bond_data['Holding currency'], current_fx_rate)
        
        # Calculate spread duration
        spread_duration = calculate_spread_duration(eur_krds, is_callable)
        
        # Calculate FX sensitivity
        fx_sensitivity = calculate_fx_sensitivity(bond_data, net_weight, current_fx_rate, base_price)
        
        # Build 9-element exposure vector
        exposure_vector = np.array([
            net_weight * eur_krds['1Y'],
            net_weight * eur_krds['2Y'],
            net_weight * eur_krds['3Y'],
            net_weight * eur_krds['4Y'],
            net_weight * eur_krds['5Y'],
            net_weight * eur_krds['7Y'],
            net_weight * eur_krds['10Y'],
            net_weight * spread_duration,
            net_weight * fx_sensitivity
        ])
        
        # Extract 9×9 covariance matrix
        factor_order = ['1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y', 'OAS', 'FX']
        cov_matrix = annual_cov[factor_order].loc[factor_order].values
        
        # Calculate total TEV
        total_variance = exposure_vector @ cov_matrix @ exposure_vector
        total_tev = np.sqrt(total_variance) if total_variance > 0 else 0
        
        # Calculate marginal TE
        if total_tev > 0:
            marginal_te = (cov_matrix @ exposure_vector) / total_tev
        else:
            marginal_te = np.zeros(9)
        
        # Calculate contributions
        contribution = exposure_vector * marginal_te
        
        # Convert to basis points
        contribution_bps = contribution * 10000
        total_tev_bps = total_tev * 10000
        
        # Aggregate into 3 main components
        curve_te_bps = np.sum(contribution_bps[0:7])
        spread_te_bps = contribution_bps[7]
        fx_te_bps = contribution_bps[8]
        
        # Build result dictionary
        result = {
            'Security': security_name,
            'Holding_Currency': bond_data['Holding currency'],
            'Portfolio_Weight_%': portfolio_weight * 100,
            'Benchmark_Weight_%': benchmark_weight * 100,
            'Net_Weight_%': net_weight * 100,
            'Is_Callable': is_callable,
            'Modified_Duration': sum(eur_krds.values()),
            'Spread_Duration': spread_duration,
            'Total_TEV_bps': total_tev_bps,
            'Curve_TE_bps': curve_te_bps,
            'Gov_Spread_TE_bps': spread_te_bps,
            'FX_TE_bps': fx_te_bps,
            # 9 Marginal TEs
            'Marginal_TE_1Y': marginal_te[0],
            'Marginal_TE_2Y': marginal_te[1],
            'Marginal_TE_3Y': marginal_te[2],
            'Marginal_TE_4Y': marginal_te[3],
            'Marginal_TE_5Y': marginal_te[4],
            'Marginal_TE_7Y': marginal_te[5],
            'Marginal_TE_10Y': marginal_te[6],
            'Marginal_TE_OAS': marginal_te[7],
            'Marginal_TE_FX': marginal_te[8],
            # 9 Contributions
            'Contribution_1Y_bps': contribution_bps[0],
            'Contribution_2Y_bps': contribution_bps[1],
            'Contribution_3Y_bps': contribution_bps[2],
            'Contribution_4Y_bps': contribution_bps[3],
            'Contribution_5Y_bps': contribution_bps[4],
            'Contribution_7Y_bps': contribution_bps[5],
            'Contribution_10Y_bps': contribution_bps[6],
            'Contribution_OAS_bps': contribution_bps[7],
            'Contribution_FX_bps': contribution_bps[8],
            # Verification
            'Sum_Contributions_bps': np.sum(contribution_bps)
        }
        
        return result
        
    except Exception as e:
        program_log.append({
            'Security': security_name,
            'Issue': 'Calculation error',
            'Action': 'Skipped',
            'Details': str(e)
        })
        return None

# =============================================================================
# STEP 9: PORTFOLIO-LEVEL PROCESSING
# =============================================================================

def process_all_securities(holdings_df, current_rates, current_oas, current_fx_rate,
                          annual_cov, calendar, portfolio_value_eur):
    """
    Process all securities and return results
    Properly handles securities appearing in both Portfolio and Benchmark
    """
    print("\n[4/12] Processing all securities...")
    
    results = []
    program_log = []
    
    # Get unique securities
    unique_securities = holdings_df['Security'].unique()
    total_securities = len(unique_securities)
    
    for idx, security_name in enumerate(unique_securities, 1):
        # Get all rows for this security
        security_rows = holdings_df[holdings_df['Security'] == security_name]
        
        # Calculate net weight from Portfolio and Benchmark rows
        portfolio_weight = 0
        benchmark_weight = 0
        
        for _, row in security_rows.iterrows():
            if row['Portfolio/Benchmark'] == 'Portfolio':
                portfolio_weight += row['Weight']
            elif row['Portfolio/Benchmark'] == 'Benchmark':
                benchmark_weight += row['Weight']
        
        # Use first row for bond characteristics (should be same for all instances)
        bond_data = security_rows.iloc[0].copy()
        
        # Override with calculated net weight
        bond_data['_portfolio_weight'] = portfolio_weight
        bond_data['_benchmark_weight'] = benchmark_weight
        
        print(f"      [{idx}/{total_securities}] {security_name}...", end='')
        
        result = calculate_security_tev(
            bond_data, current_rates, current_oas, current_fx_rate,
            annual_cov, calendar, portfolio_value_eur, program_log
        )
        
        if result:
            results.append(result)
            print(" ✓")
        else:
            print(" ✗ (see program log)")
    
    print(f"\n      Processed: {len(results)}/{total_securities} securities")
    print(f"      Skipped: {len(program_log)} securities")
    
    return pd.DataFrame(results), pd.DataFrame(program_log)

# =============================================================================
# STEP 10: VALIDATION
# =============================================================================

def create_validation_sheet(results_df):
    """
    Create validation checks
    """
    print("\n[5/12] Creating validation sheet...")
    
    validation_data = []
    
    for _, row in results_df.iterrows():
        sum_contributions = row['Sum_Contributions_bps']
        total_tev = row['Total_TEV_bps']
        diff = abs(sum_contributions - total_tev)
        
        status = 'PASS' if diff < VALIDATION_TOLERANCE_BPS else 'FAIL'
        
        validation_data.append({
            'Security': row['Security'],
            'Total_TEV_bps': total_tev,
            'Sum_Contributions_bps': sum_contributions,
            'Difference_bps': diff,
            'Status': status
        })
    
    validation_df = pd.DataFrame(validation_data)
    
    pass_count = (validation_df['Status'] == 'PASS').sum()
    fail_count = (validation_df['Status'] == 'FAIL').sum()
    
    print(f"      Validation: {pass_count} PASS, {fail_count} FAIL")
    
    return validation_df

# =============================================================================
# STEP 11: EXCEL OUTPUT
# =============================================================================

def create_excel_output(results_df, validation_df, program_log_df, annual_cov, 
                       correlation, output_file):
    """
    Create 4-sheet Excel output
    """
    print(f"\n[6/12] Creating Excel output: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: TEV Attribution Results
        # Create total row
        total_row = {
            'Security': 'TOTAL',
            'Holding_Currency': '',
            'Portfolio_Weight_%': results_df['Portfolio_Weight_%'].sum(),
            'Benchmark_Weight_%': results_df['Benchmark_Weight_%'].sum(),
            'Net_Weight_%': results_df['Net_Weight_%'].sum(),
            'Is_Callable': '',
            'Modified_Duration': '',
            'Spread_Duration': '',
            'Total_TEV_bps': results_df['Total_TEV_bps'].sum(),
            'Curve_TE_bps': results_df['Curve_TE_bps'].sum(),
            'Gov_Spread_TE_bps': results_df['Gov_Spread_TE_bps'].sum(),
            'FX_TE_bps': results_df['FX_TE_bps'].sum(),
        }
        
        # Add contribution totals
        for tenor in ['1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y', 'OAS', 'FX']:
            total_row[f'Contribution_{tenor}_bps'] = results_df[f'Contribution_{tenor}_bps'].sum()
        
        # Marginal TEs blank in total row
        for tenor in ['1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y', 'OAS', 'FX']:
            total_row[f'Marginal_TE_{tenor}'] = ''
        
        total_row['Sum_Contributions_bps'] = ''
        
        # Combine total row with results
        total_df = pd.DataFrame([total_row])
        output_df = pd.concat([total_df, results_df], ignore_index=True)
        
        output_df.to_excel(writer, sheet_name='TEV Attribution', index=False)
        
        # Sheet 2: Validation
        validation_df.to_excel(writer, sheet_name='Validation', index=False)
        
        # Sheet 3: Covariance Matrix
        annual_cov.to_excel(writer, sheet_name='Covariance Matrix')
        
        # Sheet 4: Program Log
        if len(program_log_df) > 0:
            program_log_df.to_excel(writer, sheet_name='Program Log', index=False)
        else:
            # Create empty log
            pd.DataFrame([{'Message': 'No issues encountered'}]).to_excel(
                writer, sheet_name='Program Log', index=False)
    
    print(f"      ✓ Excel file created with 4 sheets")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_enhanced_tev_attribution(rates_file, holdings_file, output_file='TEV_Attribution_Results.xlsx'):
    """
    Main function to run complete enhanced TEV attribution
    """
    print("=" * 80)
    print("ENHANCED FIXED INCOME TEV ATTRIBUTION SYSTEM")
    print("=" * 80)
    print(f"\nBase Currency: {BASE_CURRENCY}")
    print(f"Portfolio Value: €{PORTFOLIO_VALUE_EUR:,}")
    print(f"Validation Tolerance: {VALIDATION_TOLERANCE_BPS} bps")
    
    # Setup QuantLib
    calendar, ql_today = setup_quantlib()
    print(f"\nCurrent Date: {ql_today}")
    
    # Load data
    rates_df = load_treasury_rates_enhanced(rates_file)
    holdings_df = load_bond_holdings_enhanced(holdings_file)
    
    # Calculate covariance matrix
    annual_cov, annual_vol, correlation, monthly_changes = calculate_enhanced_covariance_matrix(rates_df)
    
    # Get current rates, OAS, and FX
    current_rates = rates_df[['1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y']].iloc[-1].to_dict()
    current_oas = rates_df['OAS'].iloc[-1]
    current_fx_rate = rates_df['EUR/USD'].iloc[-1]
    
    print(f"\n      Current EUR/USD: {current_fx_rate:.4f}")
    print(f"      Current OAS: {current_oas*100:.2f}%")
    
    # Process all securities
    results_df, program_log_df = process_all_securities(
        holdings_df, current_rates, current_oas, current_fx_rate,
        annual_cov, calendar, PORTFOLIO_VALUE_EUR
    )
    
    # Create validation
    validation_df = create_validation_sheet(results_df)
    
    # Create Excel output
    create_excel_output(results_df, validation_df, program_log_df, 
                       annual_cov, correlation, output_file)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✓ Total securities processed: {len(results_df)}")
    print(f"✓ Total securities skipped: {len(program_log_df)}")
    
    total_curve = results_df['Curve_TE_bps'].sum()
    total_spread = results_df['Gov_Spread_TE_bps'].sum()
    total_fx = results_df['FX_TE_bps'].sum()
    portfolio_total = total_curve + total_spread + total_fx
    
    print(f"\n✓ Portfolio-level TEV components:")
    print(f"   - Curve TE:      {total_curve:>10.2f} bps")
    print(f"   - Spread TE:     {total_spread:>10.2f} bps")
    print(f"   - FX TE:         {total_fx:>10.2f} bps")
    print(f"   - Total:         {portfolio_total:>10.2f} bps")
    
    print(f"\n✓ Output saved to: {output_file}")
    print("\n" + "=" * 80)
    print("TEV ATTRIBUTION COMPLETE")
    print("=" * 80)
    
    return results_df, validation_df, program_log_df

# =============================================================================
# EXECUTE
# =============================================================================

if __name__ == "__main__":
    # File paths - UPDATE THESE
    RATES_FILE = "C:\\Users\\amits\\Desktop\\All quant workshop\\Market Risk\\Tracking error\\Expost TE-Enhanced, Security wide TE attribution\\All_Constant_Maturity_rates.xlsx"
    HOLDINGS_FILE = "C:\\Users\\amits\\Desktop\\All quant workshop\\Market Risk\\Tracking error\\Expost TE-Enhanced, Security wide TE attribution\\Bond holdings.xlsx"
    OUTPUT_FILE = "TEV_Attribution_Results.xlsx"
    
    # Run analysis
    results, validation, program_log = run_enhanced_tev_attribution(
        RATES_FILE, 
        HOLDINGS_FILE,
        OUTPUT_FILE
    )
