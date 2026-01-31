"""
Interest Rate Models Library

A collection of stochastic interest rate models and interest rate derivative pricing models.
"""

from .vasicek import VasicekModel
from .cir import CIRModel, DiscretizationScheme
from .pricing import (
    # Path-based pricing functions (low-level)
    price_derivative,
    bond_option_price,
    caplet_price,
    floorlet_price,
    cap_price,
    floor_price,
    swaption_price,
    # Model-based pricing functions (convenient wrappers)
    swap_price,
    par_swap_rate,
    price_bond_option,
    price_caplet,
    price_floorlet,
    price_cap,
    price_floor,
    price_swaption,
    # Bond price function factories
    make_bond_price_func,
    make_vasicek_bond_price_func,
    make_cir_bond_price_func,
)

__all__ = [
    # Models
    "VasicekModel",
    "CIRModel",
    "DiscretizationScheme",
    # Path-based pricing (low-level)
    "price_derivative",
    "bond_option_price",
    "caplet_price",
    "floorlet_price",
    "cap_price",
    "floor_price",
    "swaption_price",
    # Model-based pricing (convenient)
    "swap_price",
    "par_swap_rate",
    "price_bond_option",
    "price_caplet",
    "price_floorlet",
    "price_cap",
    "price_floor",
    "price_swaption",
    # Bond price function factories
    "make_bond_price_func",
    "make_vasicek_bond_price_func",
    "make_cir_bond_price_func",
]
