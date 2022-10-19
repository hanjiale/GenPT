import numpy as np

# f1_list = [0.31206464196154915, 0.3219726893127186, 0.3166184134337001, 0.32881847865521824, 0.330316742081448]
f1_list = [0.31408330971946724, 0.324123333905383, 0.31372996406365866, 0.3431945532325119, 0.31559869036482696]
f1_array = np.array(f1_list)
f1_mean, f1_std = np.mean(f1_array), np.std(f1_array)

print(f1_mean*100, f1_std*100)