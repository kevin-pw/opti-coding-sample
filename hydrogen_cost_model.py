import numpy as np

# Create a toy hydrogen production cost model
def hydrogen_supply_cost(
        h2_demand__kg_per_hour: np.ndarray,
        electricity_price__usd_per_kwh : np.ndarray,
        electrolyzer_size__kw: float,
        electrolyzer_capacity_factor__fraction: np.ndarray
    ):
    """
    This function calculates the cost of producing hydrogen from a given amount of H2 demand and electricity price data.
    
    Parameters:
        h2_demand (numpy array, shape: (8760,1): A numpy array containing the H2 demand.
        electricity_price (numpy array, shape: (8760,1): A numpy array containing the electricity price data for Germany for each hour in 2022.
        electrolyzer_size (float): The size of the electrolyzer in kW.
        electrolyzer_capacity_factor__fraction (numpy array, shape: (8760,1): A numpy array containing the electrolyzer capacity factor for each hour.

    Returns:
        lcoh__usd_per_kg (float): The levelized cost of supplying hydrogen the demanded mass of hydrogen over the lifetime of the electrolyzer.
    """ 

    # Define parameters for the electrolyzer
    electrolyzer_capital_cost__usd_per_kw = 1500
    electrolyzer_efficiency__fraction = 0.6
    electrolyzer_lifetime__years = 20
    hydrogen_higher_heating_value__kwh_per_kg = 39.39

    # Define storage parameters
    h2_storage_capacity__kg = 1000
    h2_storage_level_start__fraction = 0.5


    # Calculate the total capital cost of the electrolyzer
    total_electrolyzer_capital_cost__usd = electrolyzer_size__kw * electrolyzer_capital_cost__usd_per_kw

    # Calculate hourly electricity consumption
    electricity_consumption__kwh_per_hour = (
        electrolyzer_size__kw
        * electrolyzer_capacity_factor__fraction
    )

    # Calculate the hourly hydrogen production mass
    h2_production__kg_per_hour = (
        electricity_consumption__kwh_per_hour
        * electrolyzer_efficiency__fraction
        / hydrogen_higher_heating_value__kwh_per_kg
    )

    ##########################
    # Storage
    ##########################
    # Determine the upper and lower storage limits based on the storage capacity and the starting storage level.
    # We want the starting storage level to always be zero (because we are working with cumulative sums)
    # so we set the lower bound storage level to be negative instead of zero.
    h2_storage_lower_bound = (
        -1 * h2_storage_capacity__kg * h2_storage_level_start__fraction
    )

    h2_storage_upper_bound = h2_storage_capacity__kg * (
        1 - h2_storage_level_start__fraction
    )

    h2_demand_supply_balance__kg_per_hour = (
        h2_production__kg_per_hour - h2_demand__kg_per_hour
    )
    
    h2_relative_storage_level__kg = bounded_cumsum(
        h2_demand_supply_balance__kg_per_hour,
        lower_bound = h2_storage_lower_bound,
        upper_bound = h2_storage_upper_bound
    )

    # We want the actual lower bound to be zero, so we move the entire curve up by the lower bound
    h2_absolute_storage_level__kg = (
        h2_relative_storage_level__kg
        - h2_storage_lower_bound
    )

    # Charge is a negative value, discharge is positive value
    h2_storage_charge_and_discharge__kg_per_hour = (
        h2_absolute_storage_level__kg
        - np.roll(h2_absolute_storage_level__kg, shift = 1)
    )

    h2_supply__kg_per_hour = np.minimum(
        h2_demand__kg_per_hour,
        h2_production__kg_per_hour - h2_storage_charge_and_discharge__kg_per_hour
    )

    #h2_deficit__kg_per_hour = np.min(
    #    h2_demand_supply_balance__kg_per_hour
    #    - h2_storage_charge_and_discharge__kg_per_hour,
    #    0
    #)

    #h2_surplus__kg_per_hour = np.max(
    #    h2_demand_supply_balance__kg_per_hour
    #    - h2_storage_charge_and_discharge__kg_per_hour,
    #    0
    #)



    # Calculate the mass of hydrogen actually supplied (i.e. production not exceeding demand)
    #h2_supply__kg_per_hour = np.minimum(h2_production__kg_per_hour, h2_demand__kg_per_hour)

    # Calculate the annual hydrogen production mass
    #annual_h2_production__kg = np.sum(h2_production__kg_per_hour)

    # Calculate the annual hydrogen supply mass
    annual_h2_supply__kg = np.sum(h2_supply__kg_per_hour)

    # Calculate the hourly hydrogen production cost
    h2_production_electricity_cost__usd_per_hour = (
        electricity_consumption__kwh_per_hour
        * electricity_price__usd_per_kwh
    )

    # Calculate the annual hydrogen production cost
    annual_h2_production_electricity_cost__usd_per_year = np.sum(h2_production_electricity_cost__usd_per_hour)

    # Calculate the total costs over the lifetime of the electrolyzer
    total_costs__usd = (
        total_electrolyzer_capital_cost__usd
        + annual_h2_production_electricity_cost__usd_per_year * electrolyzer_lifetime__years
    )

    # Calculate the total hydrogen production mass over the lifetime of the electrolyzer
    #total_h2_production__kg = annual_h2_production__kg * electrolyzer_lifetime__years

    # Calculate the total hydrogen supply mass over the lifetime of the electrolyzer
    total_h2_supply__kg = annual_h2_supply__kg * electrolyzer_lifetime__years

    # Calculate the levelized cost of hydrogen supply
    lcoh__usd_per_kg = total_costs__usd / total_h2_supply__kg

    return lcoh__usd_per_kg, h2_absolute_storage_level__kg


def bounded_cumsum(x: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    """
    Compute the cumulative sum of an array within a bounded range.
    """
    # Initialize the result array
    y = np.zeros_like(x)

    # Compute the cumulative sum
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = np.maximum(lower_bound, np.minimum(upper_bound, y[i-1] + x[i]))
    
    return y
