import numpy as np


# 保底数据

operator_pity_6x = np.zeros(81)
operator_pity_6x[1:66] = 0.008
operator_pity_6x[66:80] = 0.008 + np.arange(1, 15) * 0.05
operator_pity_6x[80] = 1

operator_pity_5x = np.zeros(11)
operator_pity_5x[1:10] = 0.08
operator_pity_5x[10] = 1

weapon_pity_6x = np.zeros(41)
weapon_pity_6x[1:40] = 0.04
weapon_pity_6x[40] = 1

weapon_pity_5x = np.zeros(11)
weapon_pity_5x[1:10] = 0.15
weapon_pity_5x[10] = 1

必出6星干员的抽数 = 80
必出5星干员的抽数 = 10
必出UP6星干员的抽数 = 120

抽到6星干员时是UP6星干员的概率 = 1/2
抽到5星干员时是UP5星干员的概率 = 1/2


必出6星武器的抽数 = 40
必出5星武器的抽数 = 10

抽到6星武器时是UP6星武器的概率 = 1/4
