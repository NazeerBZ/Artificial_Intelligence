#Write a program that takes the month (1…12) as input. 
#Print whether the season is summer, winter, spring or autumn depending upon the input month.

month = int(input());

if(month >= 3 and month <= 5 ):
    print('spring');
elif(month >=6 and month <= 8):
    print('Summer');
elif(month >= 9 and month <= 11):
    print('Autumn');
elif(month == 12 or month == 1 or month == 2):
    print('Winter');
else:
    print('Incorrect input')