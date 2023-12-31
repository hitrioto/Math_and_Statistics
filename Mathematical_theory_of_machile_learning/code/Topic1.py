# pip install ISLP
# downoad the datasets and put them into and ISL_datasets folder


# %%
import pandas as pd

Auto = pd.read_csv(r"ISL_datasets/Auto.csv")


# %%
Auto = pd.read_csv("ISL_datasets/Auto.data", delim_whitespace=True)

print(Auto)

# all the stuff in the book
# loc, iloc,

# %%
from ISLP import load_data

# %%
