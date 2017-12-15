

#iMac用
"""
latlon145_file_name = "../data/" + "latlon_ex.csv"
latlon900_file_name = "../data/IC0_csv/" + "latlon_info.csv"
grid900to145_file_name = "../data/" + "grid900to145.csv"
ocean_grid_file = "../data/ocean_grid_145.csv"
ocean_grid_145 = pd.read_csv(ocean_grid_file, header=None)
ocean_idx = np.array(ocean_grid_145[ocean_grid_145==1].dropna().index)
"""


#macbook pro用
latlon145_file_name = "../../data/" + "latlon_ex.csv"
latlon900_file_name = "../../data/" + "latlon_info.csv"
grid900to145_file_name = "../../data/" + "grid900to145.csv"
ocean_grid_file = "../../data/ocean_grid_145.csv"
ocean_grid_145 = pd.read_csv(ocean_grid_file, header=None)
ocean_idx = np.array(ocean_grid_145[ocean_grid_145==1].dropna().index)