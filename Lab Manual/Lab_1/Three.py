#To determine whether a year is a leap year, follow these steps:
#1.	If the year is evenly divisible by 4, go to step 2. Otherwise, go to step 5.
#2.	If the year is evenly divisible by 100, go to step 3. Otherwise, go to step 4.
#3.	If the year is evenly divisible by 400, go to step 4. Otherwise, go to step 5.
#4.	The year is a leap year (it has 366 days).
#5.	The year is not a leap year (it has 365 days).


#year = int(input());
#
#if (year % 4 == 0 and year % 100 == 0 and year % 400 == 0):
#       print(str(year) + ' is a leap year');
#elif(year % 4 == 0 and year % 100 != 0):
#       print(str(year) + ' is a leap year');
#else:
#    print(str(year) + ' is not a leap year');