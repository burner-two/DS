#Hypothesis Testing 
  #  • Formulate null and alternative hypotheses for a given problem.
   # • Conduct a hypothesis test using appropriate statistical tests (e.g., t-test, chi-square test).
    #• Interpret the results and draw conclusions based on the test outcomes


from scipy import stats
import numpy as np

#One Sampled T-Test

#Creating a sample of ages
ages = [45,89,23,46,12,69,45,24,34,67]
print(ages)

#Calcluatinf the mean of the sample 
mean = np.mean(ages)
print("The Mean of the ages is ",mean)

#define the null hypothesis 
H0 = "the average age of 10 people is 30"

#define the alternative hypothesis 
H1 = "the average age of 10 people is more than 30"

#performing the T-Test
t_stat,p_val = stats.ttest_1samp(ages,30)
print("P_value is: ",p_val)
print("The T-Stastic is :",t_stat)

#taking the threshold value as 0.5 or 50%
if p_val <  0.05:
    print("We can reject the  null hypothesis")
else:
    print("We can accept the null hypothesis")


#INDEPENDENT T_TEST or Two Sampled T-TEST

# Creating the data groups
data_group1 = np.array([
    12, 18, 12, 13, 15, 1, 7,
    20, 21, 25, 19, 31, 21, 17,
    17, 15, 19, 15, 12, 15
])

data_group2 = np.array([
    23, 22, 24, 25, 21, 26, 21,
    21, 25, 30, 24, 21, 23, 19,
    14, 18, 14, 12, 19, 15
])

# Calculating the mean of the two data groups
mean1 = np.mean(data_group1)
mean2 = np.mean(data_group2)

# Print mean values
print("data group 1 mean value:", mean1)
print("data group 2 mean value:", mean2)

# Calculating standard deviation
std1 = np.std(data_group1)
std2 = np.std(data_group2)

# Printing standard deviation values
print("data group 1 std value:", std1)
print("data group 2 std value:", std2)

#Define the null Hypotheisis
H2 = "Independent sample means are equal "

#Define the alternative Hypothesis 
H3 = "Independent sample means are not equal"

# Performing Independent T-Test
t_statistic, p_value = stats.ttest_ind(data_group1, data_group2)

# Displaying test results
print("\nT-Statistic:", t_statistic)
print("P-Value:", p_value)

# Interpretation
if p_value < 0.05:
    print("Result: Statistically significant difference between the two groups.")
else:
    print("Result: No statistically significant difference between the two groups.")




#Chi-Square Test

Data = [[231,256,321],[245,312,213]]

#Define the null Hypothesis
H5 = "ther is no relationship between variable"

#Define the alternative Hypo
H6 ="there is significant relationship between vairavble"

#performing the chi_square test 
t_stats ,p_vals , dof,expected_val = stats.chi2_contingency(Data)
print("the value of our test is "+ str(p_vals))

#checking the hypothesis 
if p_vals >= 0.05:
    print("we can reject  the null hypothesis")
else:
    print("we can accept the null hypothesis")
