#ANOVA 
#perform One Way anova to compare means across multiple group

from scipy import stats

performance1 = [89,89,88,78,79]
performance2 = [93,92,94,89,88]
performance3 = [89,88,89,93,90]
performance4 = [81,78,81,92,82]

#conduct  the One_Way Anova F-Test

f_stats,p_val = stats.f_oneway(performance1,performance2,performance3,performance4)

print("p_val: ",p_val)

#taking the threshold value as 0.05 or 50%

if p_val < 0.05:
    print("we can reject the null hypothesis ")
else:
    print("we can accept the null hypothesis")
