# opti-coding-sample by Kevin Palmer-Wilson
This repo contains a coding sample that demonstrates the optimization of a toy hydrogen cost model. The model optimizes electrolyzer size and electroylzer capacity factors over a one year time period at hourly resolution. The optimization is designed to minimize levelized cost of hydrogen production while supplying all hydrogen demand.

The model takes as inputs an array of 8760 electricity prices, and an array of 8760 hourly hydrogen demand values.

The model is defined in the file `hydrogen_cost_model.py`. The optimization using gradient descent is demonstrated in the jupyter notebook `gradient_descent_example.ipynb`.

