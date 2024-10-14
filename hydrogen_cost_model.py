import numpy as np
import jax
import jax.numpy as jnp

# Create a toy hydrogen production cost model
def hydrogen_supply_cost(
        h2_demand__kg_per_hour: jnp.ndarray,
        electricity_price__usd_per_kwh : jnp.ndarray,
        electrolyzer_size__mw: float,
        electrolyzer_capacity_factor__fraction: jnp.ndarray,
        h2_storage_capacity__t: float
    ):

    # Define parameters for the electrolyzer
    electrolyzer_capital_cost__usd_per_kw = 1500.
    electrolyzer_efficiency__fraction = 0.6
    electrolyzer_lifetime__years = 20.
    hydrogen_higher_heating_value__kwh_per_kg = 39.39
    h2_deficit_penalty_factor = 10 # This is a penalty on unfulfilled demand

    # Define storage parameters
    h2_storage_capacity__kg = h2_storage_capacity__t * 1000.
    h2_storage_capital_cost__usd_per_kg = 600.
    h2_storage_level_start__fraction = 0.5

    # Convert from MW to kW
    electrolyzer_size__kw = electrolyzer_size__mw * 1000.

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

    # Calculate the capital cost of storage based on its capacity
    total_h2_storage_capital_cost__usd = h2_storage_capacity__kg * h2_storage_capital_cost__usd_per_kg

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
    
    # Shift the demand supply balance by one time step to enable calculating the storage level after a charging or discharging event.
    h2_demand_supply_balance_shifted__kg_per_hour = jnp.roll(h2_demand_supply_balance__kg_per_hour, shift=1)

    h2_relative_storage_level__kg = bounded_cumsum(
        h2_demand_supply_balance_shifted__kg_per_hour,
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
        jnp.roll(h2_absolute_storage_level__kg, shift = -1)
        - h2_absolute_storage_level__kg
    )

    # h2 supply cannot exceed the demand
    h2_supply__kg_per_hour = jnp.minimum(
        h2_demand__kg_per_hour,
        h2_production__kg_per_hour - h2_storage_charge_and_discharge__kg_per_hour
    )

    # Calculate any unfulfilled demand
    h2_deficit__kg_per_hour = jnp.minimum(
        h2_demand_supply_balance__kg_per_hour
        - h2_storage_charge_and_discharge__kg_per_hour,
        0
    )

    # Calculate any surplus production
    h2_surplus__kg_per_hour = jnp.maximum(
        h2_demand_supply_balance__kg_per_hour
        - h2_storage_charge_and_discharge__kg_per_hour,
        0
    )

    # Calculate annual demand
    annual_h2_demand__kg_per_year = jnp.sum(h2_demand__kg_per_hour)

    # Calculate annual deficit ration used to penalize any unfulfilled demand
    annual_h2_deficit__kg_per_year = jnp.sum(h2_deficit__kg_per_hour)
    annual_h2_deficit__ratio = -annual_h2_deficit__kg_per_year / annual_h2_demand__kg_per_year

    # Calculate the annual hydrogen supply mass
    annual_h2_supply__kg = jnp.sum(h2_supply__kg_per_hour)

    # Calculate the hourly hydrogen production cost
    h2_production_electricity_cost__usd_per_hour = (
        electricity_consumption__kwh_per_hour
        * electricity_price__usd_per_kwh
    )

    # Calculate the annual hydrogen production cost
    annual_h2_production_electricity_cost__usd_per_year = jnp.sum(h2_production_electricity_cost__usd_per_hour)

    # Calculate the total costs over the lifetime of the electrolyzer
    total_costs__usd = (
        total_electrolyzer_capital_cost__usd
        + annual_h2_production_electricity_cost__usd_per_year * electrolyzer_lifetime__years
        + total_h2_storage_capital_cost__usd
    )

    # Calculate the total hydrogen supply mass over the lifetime of the electrolyzer
    total_h2_supply__kg = annual_h2_supply__kg * electrolyzer_lifetime__years

    # Calculate the levelized cost of hydrogen supply
    lcoh__usd_per_kg = total_costs__usd / total_h2_supply__kg

    # Calculate the LCOH with penalty for unfulfilled demand
    lcoh_with_deficit_penalty__usd_per_kg = lcoh__usd_per_kg * ( 1 + annual_h2_deficit__ratio * h2_deficit_penalty_factor)

    return (
        lcoh_with_deficit_penalty__usd_per_kg,
        h2_absolute_storage_level__kg,
        annual_h2_deficit__ratio,
        h2_deficit__kg_per_hour,
        h2_surplus__kg_per_hour,
        h2_production__kg_per_hour,
        h2_storage_charge_and_discharge__kg_per_hour
    )


# Helper function to calculate the cumulative sum of an array with a lower and upper bound
def bounded_cumsum(x, lower_bound, upper_bound):
    def body(carry, xi):
        sum_so_far = carry
        new_sum = jnp.clip(sum_so_far + xi, lower_bound, upper_bound)
        return new_sum, new_sum

    initial_carry = 0.0  # Initial sum
    _, y = jax.lax.scan(body, initial_carry, x)
    return y